# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Visual-Text Transformations or Augmentations."""

import re
from typing import Any, Dict, Optional, Protocol, cast

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils import log
from cosmos_framework.data.generator.augmentors.reasoner.sample_data_stats import (
    SampleDataStats,
    system_prompt_from,
)
from cosmos_framework.data.generator.reasoner.video_decoder_qwen import (
    VideoTemporalMode,
    get_effective_temporal_patch_size,
    token_to_pixels,
)
from cosmos_framework.data.generator.processors import build_audio_processor
from cosmos_framework.data.generator.processors.audio_utils import (
    AUDIO_END_TOKEN,
    AUDIO_PAD_TOKEN,
    AUDIO_START_TOKEN,
    DEFAULT_REASONER_VIDEO_FPS,
    AudioSpecialTokens,
    add_reasoner_audio_special_tokens,
    expand_audio_placeholders_in_text,
    get_audio_only_timestamps,
    get_audio_segment_token_lengths,
    get_qwen_video_timestamps,
    splice_audio_segments_after_video_chunks,
)
from cosmos_framework.data.generator.processors.qwen3vl_processor import VIDEO_BUDGET_RESIZED_KEY
from cosmos_framework.data.generator.processors.qwen3vl_processor import Qwen3VLProcessor as Processor
from cosmos_framework.utils.generator.reasoner.constant import IGNORE_INDEX, MEDIA_ROLES, PROCESSOR_KEYS_TO_ADD
from cosmos_framework.utils.generator.source_video_timing import (
    SOURCE_VIDEO_TIMING_KEY,
    require_source_pts_processor,
    validate_source_video_timing,
    validate_video_timestamp_mode,
)
from cosmos_framework.utils.generator.video_source_metadata import VIDEO_METADATA_KEY, validate_source_video_metadata


class _AudioProcessor(Protocol):
    sampling_rate: int

    def __call__(
        self,
        audios: Any,
        *,
        sampling_rate: int,
    ) -> dict[str, torch.Tensor]: ...

    def get_token_timestamps(self, audio_feature_length: int) -> list[float]: ...


def expand_video_frames_for_framewise(
    videos: list,
    frames_indices: list[int],
    temporal_patch_size: int,
) -> tuple[list, list[int]]:
    """Repeat each source frame across the pretrained temporal tubelet."""
    if temporal_patch_size <= 0:
        raise ValueError(f"temporal_patch_size must be positive, got {temporal_patch_size}")
    if len(videos) != len(frames_indices):
        raise ValueError(
            f"Expected one source index per video frame, got {len(frames_indices)} indices for {len(videos)} frames"
        )
    expanded_videos = [frame for frame in videos for _ in range(temporal_patch_size)]
    expanded_indices = [index for index in frames_indices for _ in range(temporal_patch_size)]
    return expanded_videos, expanded_indices


def _select_video_frames(content: dict[str, Any], indices: list[int]) -> None:
    """Select pixels and their timing metadata together, without mutating source records."""
    num_frames = len(content["video"])
    if "frames_indices" in content:
        if len(content["frames_indices"]) != num_frames:
            raise ValueError("Framewise metadata must have one source index per video frame")
        content["frames_indices"] = [content["frames_indices"][index] for index in indices]
    if VIDEO_METADATA_KEY in content:
        metadata = validate_source_video_metadata(content[VIDEO_METADATA_KEY], num_frames)
        metadata["frames_indices"] = [metadata["frames_indices"][index] for index in indices]
        content[VIDEO_METADATA_KEY] = metadata
    if SOURCE_VIDEO_TIMING_KEY in content:
        timing = content[SOURCE_VIDEO_TIMING_KEY]
        validate_source_video_timing(timing, num_frames)
        selected_timing = dict(timing)
        for key in ("frame_indices", "pts_seconds", "duration_seconds"):
            selected_timing[key] = [timing[key][index] for index in indices]
        selected_timing["origin_pts_seconds"] = selected_timing["pts_seconds"][0]
        content[SOURCE_VIDEO_TIMING_KEY] = selected_timing
    content["video"] = [content["video"][index] for index in indices]


def maybe_subsample_frames(model_name_or_path, list_of_pil_image, max_video_token_length, processor):
    """
    Why do we need to subsample frames? For model like eagle_er, it does not support smart downsampling in the processor.
    And all the frames are resized to the same size. There are 2 senerios the context length can easily exceed the limit.
    1: the video has >32 frames, it will create 256*32=8192 tokens which exceeds the limit.
    2: there are multiple images, by default, each image will be tiled into (at most) 13 tiles. Each tile is 256 tokens.
    So if there are multiple images, or many frames in the video, we need to subsample the frames to shorten the context length.
    """
    if "Qwen/Qwen2.5-VL" in model_name_or_path:
        return list_of_pil_image
    elif "eagle_er" in model_name_or_path or "InternVL3_5" in model_name_or_path:
        tokens_per_tile = processor.tokens_per_tile
        # 1 frames map to 256 tokens
        estimated_num_frames = max_video_token_length // tokens_per_tile
        if len(list_of_pil_image) > estimated_num_frames:
            # Evenly sample frames
            sample_idx = np.linspace(0, len(list_of_pil_image) - 1, estimated_num_frames).astype(int)
            return [list_of_pil_image[i] for i in sample_idx]
        else:
            return list_of_pil_image
    else:
        return list_of_pil_image


def convert_all_images_to_rgb(conversation):
    """
    Convert all images to RGB. Otherwise the tokenizer will raise error for image in LA mode.
    """
    new_conversation = []
    for conversation_round in conversation:
        if isinstance(conversation_round["content"], list):
            new_content_list = []
            for content in conversation_round["content"]:
                if "type" not in content:
                    log.critical(
                        f"content: {content} | conversation_round: {conversation_round} | full conversation: {conversation}"
                    )
                    content = {"type": "text", "text": content}
                content_type = content["type"]
                if content_type in ["image", "video"]:
                    if isinstance(content[content_type], Image.Image):
                        content[content_type] = content[content_type].convert("RGB")
                    elif isinstance(content[content_type], list):
                        content_i = content[content_type]
                        new_content_i = []
                        for img in content_i:
                            if isinstance(img, Image.Image):
                                img = img.convert("RGB")
                            new_content_i.append(img)
                        content[content_type] = new_content_i
                new_content_list.append(content)
            conversation_round["content"] = new_content_list
        new_conversation.append(conversation_round)

    return new_conversation


def compress_repeated_tokens(dialog_str):
    pattern = re.compile(r"((<\|[^|]+\|>|<｜[^<>]+｜>|\[[^\]]+\]))\1+")

    def replacer(match):
        token = match.group(1)
        count = len(match.group(0)) // len(token)
        return f"{token}*{count}times"

    # Cap length to avoid regex hang on very long decoded sequences
    max_len = 16 * 1024
    if len(dialog_str) > max_len:
        dialog_str = dialog_str[:max_len] + "...[truncated]"
    return pattern.sub(replacer, dialog_str)


class TokenizeData(Augmentor):
    """
    Image-Text Transform for Supervised Fine-Tuning (SFT) data, for Vision-Language Model training.
    """

    def __init__(
        self,
        processor: Optional[Processor] = None,
        min_video_token_length: int = 16,
        max_video_token_length: int = 8192,
        max_video_frames: int | None = None,
        max_image_token_length: int = 8192,
        max_total_token_length: int | None = None,
        adaptive_video_token_budget: bool = False,
        adaptive_video_token_safety_margin: int = 128,
        custom_system_prompt: str | None = None,
        strip_original_system_prompt: bool = False,
        data_stats_writer_enabled: bool = False,
        text_only: bool = False,
        sound_und: bool = False,
        audio_processor: Optional[_AudioProcessor] = None,
        audio_encoder_type: str = "parakeet",
        audio_start_token: str = AUDIO_START_TOKEN,
        audio_pad_token: str = AUDIO_PAD_TOKEN,
        audio_end_token: str = AUDIO_END_TOKEN,
        audio_timestamp_fps: float = DEFAULT_REASONER_VIDEO_FPS,
        audio_layout: str = "separate_with_timestamps",
        video_timestamp_mode: str = "qwen_index",
        video_temporal_mode: VideoTemporalMode = "native",
        add_vision_id: bool = False,
        keep_raw_media: bool = True,
    ) -> None:
        """
        Args:
            processor (Processor): Text/Image processor for tokenization.
            max_video_token_length (int): Maximum number of video tokens to use. Defaults to 8192.
            custom_system_prompt: Prompt to inject when no leading system message exists.
            strip_original_system_prompt: Remove existing system messages before optional injection.
            data_stats_writer_enabled: Collect per-sample metadata for DataStatsWriterCallback.
            sound_und (bool): Opt in to audio preprocessing and audio-token registration.
                Disabled by default so existing text/vision tokenizers are unchanged.
            add_vision_id (bool): Label media in samples with several images or several videos
                ("Picture N: " / "Video N: "). Off by default: the processor wrapper used to drop this flag, so
                checkpoints trained so far never saw the labels, and serving stacks do not add them by default.
                It applies to every dataset this augmentor tokenizes, and datasets whose prompts already
                interleave "Picture N" text would get duplicate labels, so it is not a recipe-wide setting.
            keep_raw_media (bool): Also emit the debug-only "raw_image" / "raw_video" uint8 tensors. Disable for
                long-context runs, where they duplicate every decoded frame in the dataloader output.
        """
        validate_video_timestamp_mode(video_timestamp_mode)
        self.video_timestamp_mode: str = video_timestamp_mode
        if video_timestamp_mode == "source_pts":
            require_source_pts_processor(processor)
            if sound_und or audio_processor is not None:
                raise ValueError("source_pts currently supports video without audio understanding")
        # Create the tokenizer
        self.text_only = text_only
        self.processor = processor  # Expecting a ImageTextTokenizer
        self.min_video_token_length = min_video_token_length
        self.max_video_token_length = max_video_token_length
        self.max_video_frames = max_video_frames
        self.max_image_token_length = max_image_token_length
        self.max_total_token_length = max_total_token_length
        if not isinstance(adaptive_video_token_budget, bool):
            raise TypeError(
                f"adaptive_video_token_budget must be a bool, got {type(adaptive_video_token_budget).__name__}"
            )
        if min_video_token_length <= 0:
            raise ValueError(f"min_video_token_length must be positive, got {min_video_token_length}")
        if max_video_token_length < min_video_token_length:
            raise ValueError(
                "max_video_token_length must be greater than or equal to min_video_token_length, "
                f"got min={min_video_token_length}, max={max_video_token_length}"
            )
        if max_video_frames is not None and max_video_frames <= 0:
            raise ValueError(f"max_video_frames must be positive, got {max_video_frames}")
        if adaptive_video_token_safety_margin < 0:
            raise ValueError(
                f"adaptive_video_token_safety_margin must be non-negative, got {adaptive_video_token_safety_margin}"
            )
        if adaptive_video_token_budget and (
            max_total_token_length is None or max_total_token_length <= adaptive_video_token_safety_margin
        ):
            raise ValueError(
                "adaptive_video_token_budget requires max_total_token_length greater than "
                f"adaptive_video_token_safety_margin, got max_total_token_length={max_total_token_length}, "
                f"margin={adaptive_video_token_safety_margin}"
            )
        self.adaptive_video_token_budget = adaptive_video_token_budget
        self.adaptive_video_token_safety_margin = adaptive_video_token_safety_margin
        self.custom_system_prompt = custom_system_prompt
        self.strip_original_system_prompt = strip_original_system_prompt
        self.collect_data_stats = data_stats_writer_enabled
        self.add_vision_id = add_vision_id
        self.keep_raw_media = keep_raw_media
        self.video_temporal_mode: VideoTemporalMode = video_temporal_mode
        self.effective_temporal_patch_size: int = get_effective_temporal_patch_size(
            getattr(self.processor, "temporal_patch_size", 2), video_temporal_mode
        )
        if sound_und and video_temporal_mode == "framewise":
            raise ValueError("Framewise video with sound_und requires a shared audio/video timestamp clock")
        if not isinstance(sound_und, bool):
            raise TypeError(f"sound_und must be a bool, got {type(sound_und).__name__}")
        if not sound_und and audio_processor is not None:
            raise ValueError("audio_processor requires sound_und=True")

        self.sound_und = sound_und
        self.audio_processor = None
        self.audio_special_tokens: AudioSpecialTokens | None = None
        self.audio_timestamp_fps = audio_timestamp_fps
        self.audio_layout = audio_layout
        if sound_und:
            if audio_processor is not None:
                self.audio_processor = audio_processor
            else:
                self.audio_processor = build_audio_processor(audio_encoder_type)
            self.audio_special_tokens = add_reasoner_audio_special_tokens(
                self.processor.tokenizer,
                model_name_or_path=self.processor.name,
                audio_start_token=audio_start_token,
                audio_pad_token=audio_pad_token,
                audio_end_token=audio_end_token,
            )

    def _video_token_counts(self, tokenizer_output: dict[str, torch.Tensor]) -> list[int]:
        """Count post-merge tokens for each video represented by video_grid_thw."""
        video_grid_thw = tokenizer_output.get("video_grid_thw")
        if video_grid_thw is None:
            return []
        grid = torch.as_tensor(video_grid_thw, dtype=torch.int64)
        if grid.numel() == 0:
            return []
        if grid.ndim == 1:
            grid = grid.reshape(1, -1)
        unmerged_tokens = grid.prod(dim=-1)
        merge_area = int(self.processor.merge_size) ** 2
        if bool(torch.any(unmerged_tokens % merge_area != 0)):
            raise ValueError(
                f"video_grid_thw contains per-video counts {unmerged_tokens.tolist()}, not all divisible by "
                f"merge_size**2={merge_area}"
            )
        return [int(value) for value in (unmerged_tokens // merge_area).tolist()]

    def _count_video_tokens(self, tokenizer_output: dict[str, torch.Tensor]) -> int:
        """Count aggregate post-merge video tokens represented by video_grid_thw."""
        return sum(self._video_token_counts(tokenizer_output))

    def _apply_video_token_budget(
        self,
        conversation: list[dict[str, Any]],
        video_token_budget: int,
        *,
        allow_frame_subsampling: bool,
    ) -> bool:
        """Apply one aggregate video budget across all videos in a conversation."""
        video_contents = [
            content
            for message in conversation
            if isinstance(message.get("content"), list)
            for content in message["content"]
            if content.get("type") == "video"
        ]
        if not video_contents:
            return False

        patch_size = int(self.processor.patch_size)
        temporal_patch_size = int(self.processor.temporal_patch_size)
        merge_size = int(self.processor.merge_size)
        total_pixel_budget = token_to_pixels(
            video_token_budget,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
        )
        pixel_budget_per_video = total_pixel_budget // len(video_contents)
        min_pixels_per_frame = self._minimum_video_pixels_per_frame()

        for content in video_contents:
            frames = content["video"]
            if allow_frame_subsampling:
                max_frames = pixel_budget_per_video // min_pixels_per_frame
                max_frames = max_frames // temporal_patch_size * temporal_patch_size
                if max_frames < temporal_patch_size:
                    return False
                if len(frames) > max_frames:
                    if self.video_temporal_mode == "framewise":
                        # Keep the repeated tubelet for each selected source frame intact.
                        groups = np.linspace(
                            0, len(frames) // temporal_patch_size - 1, max_frames // temporal_patch_size
                        ).astype(int)
                        sample_idx = [
                            int(group) * temporal_patch_size + offset
                            for group in groups
                            for offset in range(temporal_patch_size)
                        ]
                    else:
                        sample_idx = np.linspace(0, len(frames) - 1, max_frames).astype(int).tolist()
                    _select_video_frames(content, sample_idx)
                    frames = content["video"]

            max_pixels_per_frame = pixel_budget_per_video // len(frames)
            if max_pixels_per_frame < min_pixels_per_frame:
                return False
            content["max_pixels"] = max_pixels_per_frame
            if self.processor.use_smart_resize:
                resized_frames = self._resize_video_frames(frames, max_pixels_per_frame)
                if resized_frames is None:
                    return False
                content["video"] = resized_frames
        return True

    def _minimum_video_pixels_per_frame(self) -> int:
        """Return the model-legal video floor without changing the decoder's training-quality floor."""
        factor = int(self.processor.patch_size) * int(self.processor.merge_size)
        wrapped_processor = getattr(self.processor, "processor", None)
        video_processor = getattr(wrapped_processor, "video_processor", None)
        video_size = getattr(video_processor, "size", {})
        shortest_edge = video_size.get("shortest_edge") if isinstance(video_size, dict) else None
        if shortest_edge is not None:
            return max(int(shortest_edge), factor**2)
        min_height_width = int(getattr(self.processor, "min_height_width", factor))
        return max(min_height_width**2, factor**2)

    def _resize_video_frames(
        self,
        frames: list[Image.Image],
        max_pixels_per_frame: int,
    ) -> list[Image.Image] | None:
        """Downscale frames so processors that ignore content max_pixels still honor the budget."""
        patch_size = int(self.processor.patch_size)
        merge_size = int(self.processor.merge_size)
        factor = patch_size * merge_size
        min_pixels_per_frame = self._minimum_video_pixels_per_frame()
        if max_pixels_per_frame < min_pixels_per_frame:
            return None

        resized_frames: list[Image.Image] = []
        for frame in frames:
            width, height = frame.size
            if width * height <= max_pixels_per_frame:
                resized_frames.append(frame)
                continue
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=factor,
                min_pixels=min_pixels_per_frame,
                max_pixels=max_pixels_per_frame,
            )
            resized_frames.append(frame.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC))
        return resized_frames

    def __call__(self, data_dict: Dict) -> Dict:
        r"""Tokenize a dialog and pad the sequence.

        "media" is a dict of
        {
            "video_1": {"video": [PIL.Image.Image, ...], "fps": int},
            "image_1": PIL.Image.Image,
            "audio_1": np.ndarray | torch.Tensor,  # mono floating-point waveform at 16 kHz
        }

        "conversation" is a list of dicts, each dict has the following fields:
        {
            "role": "user" or "assistant",
            "content": [
                {"type": "video", "video": media_key_in_media_dict},
                {"type": "image", "image": media_key_in_media_dict},
                {"type": "audio", "audio": media_key_in_media_dict},
                {"type": "text", "text": str},
            ],
        }
        or
        {
            "role": "user" or "assistant",
            "content": str,
        }

        Args:
            data_dict (dict): Input data dict

        Returns:
            data_dict (dict): Output dict
        """
        conversation = data_dict["conversation"]
        if self.video_timestamp_mode == "source_pts":
            for message in conversation:
                contents = message.get("content")
                if isinstance(contents, list) and any(content.get("type") == "audio" for content in contents):
                    raise ValueError("source_pts does not support audio conversation content")
            for value in data_dict.get("media", {}).values():
                if isinstance(value, dict) and "audio" in value:
                    raise ValueError("source_pts does not support paired audio media")
        processor_kwargs = {}
        total_images = 0
        total_videos = 0
        total_audios = 0
        raw_images: list[torch.Tensor] = []
        raw_videos: list[torch.Tensor] = []
        audio_clips: list[np.ndarray | torch.Tensor] = []
        audio_start_seconds: list[float] = []
        # Pre-compute modality counts. Audio clips follow the same typed-content
        # -> media-dictionary schema as images and videos.
        for message in conversation:
            if not isinstance(message, dict):
                raise ValueError(
                    f"message is not a dict: {message} | conversation: {conversation} | data_dict: {data_dict} | __url__: {data_dict['__url__'].root}, {data_dict['__url__'].path}"
                )
            if message["role"] in MEDIA_ROLES and isinstance(message["content"], list):
                total_images += len([content for content in message["content"] if content["type"] == "image"])
                total_videos += len([content for content in message["content"] if content["type"] == "video"])
                total_audios += len([content for content in message["content"] if content["type"] == "audio"])
        data_stats = SampleDataStats.from_conversation(conversation) if self.collect_data_stats else None

        # url
        url = data_dict["__url__"].root + "/" + data_dict["__url__"].path

        audio_outputs: dict[str, torch.Tensor] | None = None
        if total_audios > 0:
            if self.audio_processor is None or self.audio_special_tokens is None:
                log.critical(
                    f"[TokenizerDataError]audio content requires an audio_processor. url: {url}",
                    rank0_only=False,
                )
                return None
            if "media" not in data_dict:
                log.critical(
                    f"[TokenizerDataError]media not found for audio content. url: {url}",
                    rank0_only=False,
                )
                return None

            for message in conversation:
                if message["role"] not in MEDIA_ROLES or not isinstance(message["content"], list):
                    continue
                for content in message["content"]:
                    if content["type"] != "audio":
                        continue
                    media_key = content.get("audio")
                    if media_key not in data_dict["media"]:
                        log.critical(
                            f"[TokenizerDataError]audio {media_key!r} not found in media, "
                            f"available keys: {data_dict['media'].keys()}. url: {url}",
                            rank0_only=False,
                        )
                        return None
                    audio = data_dict["media"][media_key]
                    audio_start_seconds.append(
                        float(audio.get("audio_start_seconds", 0.0)) if isinstance(audio, dict) else 0.0
                    )
                    if isinstance(audio, dict):
                        if "audio" not in audio:
                            log.critical(
                                f"[TokenizerDataError]audio stream not found in media[{media_key!r}]. url: {url}",
                                rank0_only=False,
                            )
                            return None
                        audio = audio["audio"]
                    audio_clips.append(audio)

            try:
                audio_outputs = self.audio_processor(
                    audio_clips,
                    sampling_rate=self.audio_processor.sampling_rate,
                )
            except (TypeError, ValueError) as e:
                log.critical(f"[TokenizerDataError]audio preprocessing failed: {e}. url: {url}", rank0_only=False)
                return None

            if audio_outputs["audio_token_lengths"].shape != (total_audios,):
                log.critical(
                    "[TokenizerDataError]audio processor returned one token length per clip; "
                    f"expected {total_audios}, got {tuple(audio_outputs['audio_token_lengths'].shape)}. url: {url}",
                    rank0_only=False,
                )
                return None

        # go through each message in the conversation
        audio_index = 0
        audio_segment_lengths_by_video: list[list[int] | None] = []
        for message in conversation:
            video_timestamps_by_key: dict[str, list[float]] = {}
            active_video_key: str | None = None
            # for user and tool messages, we insert the media
            if message["role"] in MEDIA_ROLES and isinstance(
                message["content"], list
            ):  # Otherwise it's text and content is a string
                images_content_idx_full = [
                    content_idx for content_idx, content in enumerate(message["content"]) if content["type"] == "image"
                ]
                images_content_idx_subsampled = maybe_subsample_frames(
                    self.processor.name, images_content_idx_full, self.max_image_token_length, self.processor
                )
                if (
                    len(images_content_idx_subsampled) > 0
                ):  # for eagle, we need to reduce the max_dynamic_tiles and not use thumbnail. These args only valid for eagle_er processor.
                    processor_kwargs["max_dynamic_tiles"] = 1
                    processor_kwargs["use_thumbnail"] = False

                message_has_video = any(content["type"] == "video" for content in message["content"])
                message_has_audio = any(content["type"] == "audio" for content in message["content"])
                new_content_list = []
                for content_idx, content in enumerate(message["content"]):
                    if content["type"] == "image":
                        if content_idx not in images_content_idx_subsampled:
                            continue
                        # for image, we do NOT use the temporal patch size, this leads to a smaller max_pixels
                        # Later, each image will be repeated temporal_patch_size times
                        max_total_pixels = token_to_pixels(
                            self.max_image_token_length,
                            patch_size=self.processor.patch_size,
                            temporal_patch_size=1,  # Because this is image, not video
                        )
                        max_pixels_per_image = max_total_pixels // total_images

                        if self.processor.use_smart_resize:
                            min_pixels_per_image = self.processor.processor.image_processor.size["shortest_edge"]
                            if max_pixels_per_image < min_pixels_per_image:
                                log.critical(
                                    f"max_pixels_per_image: {max_pixels_per_image} < min_pixels_per_image: {min_pixels_per_image} | self.max_image_token_length = {self.max_image_token_length} is not enough for total_images: {total_images}, as the default min_pixels is {min_pixels_per_image} | Either increase max_image_token_length or include max_pixels in the content or reduce min_pixels"
                                )
                                return None

                        # Add each image to the content list
                        if "media" not in data_dict:
                            log.critical(
                                f"[TokenizerDataError]media not found in data_dict, available keys: {data_dict.keys()}. url: {url}, content: {message['content']}",
                                rank0_only=False,
                            )
                            return None

                        elif content["image"] not in data_dict["media"]:
                            log.critical(
                                f"[TokenizerDataError]image {content['image']} not found in media, available keys: {data_dict['media'].keys()}. url: {url}",
                                rank0_only=False,
                            )
                            return None
                        image = data_dict["media"][content["image"]]
                        content["image"] = image
                        content["max_pixels"] = max_pixels_per_image
                        native_width, native_height = image.size
                        if self.keep_raw_media:
                            raw_image = np.asarray(image.convert("RGB"))  # [H,W,3]
                            raw_images.append(torch.from_numpy(raw_image).permute(2, 0, 1)[:, None])  # [3,1,H,W]
                        if data_stats is not None:
                            data_stats.add_image(native_height, native_width, self.processor)

                    elif content["type"] == "video":
                        # as tokenization will NOT upsample the video, we can use a larger value here at the cost of multiple video having 1.5x token length
                        max_total_pixels = token_to_pixels(
                            self.max_video_token_length * 1.5,
                            patch_size=self.processor.patch_size,
                            temporal_patch_size=self.effective_temporal_patch_size,
                            merge_size=self.processor.merge_size,
                        )
                        media_key = content["video"]
                        # Add each video to the content list
                        if "media" not in data_dict:
                            log.critical(
                                f"[TokenizerDataError]media not found in data_dict, available keys: {data_dict.keys()}. url: {url}, content: {message['content']}",
                                rank0_only=False,
                            )
                            return None
                        if media_key not in data_dict["media"]:
                            log.info(
                                f"[TokenizerDataError]video {media_key} not found in media, available keys: {data_dict['media'].keys()}. url: {url}"
                            )
                            return None
                        video_media = data_dict["media"][media_key]
                        if "videos" not in video_media:
                            log.info(
                                f"[TokenizerDataError]videos not found in media[{media_key}], available keys: {video_media.keys()}. url: {url}"
                            )
                            return None
                        videos = video_media["videos"]  # list of PIL images
                        fps = video_media["fps"]
                        source_timing = video_media.get(SOURCE_VIDEO_TIMING_KEY)
                        if self.video_timestamp_mode == "source_pts":
                            validate_source_video_timing(source_timing, len(videos))
                        elif source_timing is not None:
                            raise ValueError(
                                "source_pts media requires TokenizeData(video_timestamp_mode='source_pts')"
                            )
                        video_metadata = video_media.get(VIDEO_METADATA_KEY)
                        if VIDEO_METADATA_KEY in video_media:
                            if self.video_timestamp_mode != "qwen_index":
                                raise ValueError(
                                    "video_metadata requires TokenizeData(video_timestamp_mode='qwen_index')"
                                )
                            video_metadata = validate_source_video_metadata(video_metadata, len(videos))
                        if data_stats is not None:
                            data_stats.add_video(
                                video_media,
                                self.processor,
                                self.effective_temporal_patch_size,
                            )

                        content["video"] = videos
                        if video_metadata is not None:
                            content[VIDEO_METADATA_KEY] = video_metadata
                        if source_timing is not None:
                            content[SOURCE_VIDEO_TIMING_KEY] = source_timing
                        if self.video_temporal_mode == "framewise":
                            content["frames_indices"] = (
                                video_metadata["frames_indices"]
                                if video_metadata is not None
                                else video_media.get("source_frames_indices", list(range(len(videos))))
                            )
                        selected_indices = maybe_subsample_frames(
                            self.processor.name, list(range(len(videos))), self.max_video_token_length, self.processor
                        )
                        if self.max_video_frames is not None and len(selected_indices) > self.max_video_frames:
                            # Framewise frames are expanded to whole tubelets later, so each source frame counts.
                            temporal_patch_size = self.effective_temporal_patch_size
                            aligned_max_video_frames = (
                                self.max_video_frames // temporal_patch_size * temporal_patch_size
                            )
                            if aligned_max_video_frames < temporal_patch_size:
                                log.critical(
                                    "[TokenizerDataError]max_video_frames must allow at least "
                                    f"{temporal_patch_size} temporally aligned frames, "
                                    f"got {self.max_video_frames}. url: {url}",
                                    rank0_only=False,
                                )
                                return None
                            sample_idx = np.linspace(0, len(selected_indices) - 1, aligned_max_video_frames).astype(int)
                            selected_indices = [selected_indices[index] for index in sample_idx]
                        if len(selected_indices) != len(videos):
                            _select_video_frames(content, selected_indices)
                            videos = content["video"]
                            video_metadata = content.get(VIDEO_METADATA_KEY)
                            source_timing = content.get(SOURCE_VIDEO_TIMING_KEY)
                        if len(videos) == 0:
                            log.info(f"[TokenizerDataError]video {media_key} has no decoded frames. url: {url}")
                            return None
                        if self.video_timestamp_mode == "source_pts":
                            validate_source_video_timing(source_timing, len(videos))
                            content[SOURCE_VIDEO_TIMING_KEY] = source_timing
                        content["video"] = videos
                        if video_metadata is not None:
                            # Any later temporal selection must update source indices with the pixels.
                            content[VIDEO_METADATA_KEY] = validate_source_video_metadata(video_metadata, len(videos))
                        content[VIDEO_BUDGET_RESIZED_KEY] = video_media.get(VIDEO_BUDGET_RESIZED_KEY, False)

                        max_pixels_per_image = max_total_pixels // total_videos // len(videos)
                        # Explicit adaptive budgeting owns this pixel change. Ordinary
                        # loaders keep the original decoded frames and processor path.
                        if self.adaptive_video_token_budget and self.processor.use_smart_resize:
                            resized_videos = self._resize_video_frames(videos, int(max_pixels_per_image))
                            if resized_videos is None:
                                log.critical(
                                    "[TokenizerDataError]video pixel budget is below the processor's minimum "
                                    f"frame geometry: max_pixels_per_image={max_pixels_per_image}. url: {url}",
                                    rank0_only=False,
                                )
                                return None
                            videos = resized_videos
                        if self.video_temporal_mode == "framewise":
                            source_indices = content["frames_indices"]
                            if len(source_indices) != len(videos):
                                log.critical(
                                    f"[TokenizerDataError]video frame metadata length {len(source_indices)} does not "
                                    f"match decoded frames {len(videos)}. url: {url}"
                                )
                                return None
                            content["video"], content["frames_indices"] = expand_video_frames_for_framewise(
                                videos, source_indices, self.processor.temporal_patch_size
                            )
                            if video_metadata is not None:
                                # Repeat the source indices with the pixels, keeping source crop offsets.
                                content[VIDEO_METADATA_KEY] = validate_source_video_metadata(
                                    {**video_metadata, "frames_indices": content["frames_indices"]},
                                    len(content["video"]),
                                )
                                content["fps"] = video_metadata["fps"]
                                content["total_num_frames"] = video_metadata["total_num_frames"]
                            else:
                                content["fps"] = video_media.get("source_fps", fps)
                                content["total_num_frames"] = video_media.get("source_total_num_frames", len(videos))
                        else:
                            content["video"] = videos
                            content["fps"] = fps
                        if message_has_audio:
                            video_timestamps_by_key[media_key] = get_qwen_video_timestamps(
                                num_frames=len(videos),
                                fps=video_metadata["fps"] if video_metadata is not None else fps,
                                temporal_patch_size=self.processor.temporal_patch_size,
                                frame_indices=video_metadata["frames_indices"] if video_metadata is not None else None,
                            )
                            active_video_key = media_key
                        if self.audio_layout == "interleaved_av":
                            audio_segment_lengths_by_video.append(None)

                        if self.keep_raw_media:
                            raw_video_frames = np.stack(
                                [np.asarray(frame.convert("RGB")) for frame in videos], axis=0
                            )  # [T,H,W,3]
                            raw_videos.append(torch.from_numpy(raw_video_frames).permute(3, 0, 1, 2))  # [3,T,H,W]
                    elif content["type"] == "audio":
                        assert audio_outputs is not None
                        assert self.audio_special_tokens is not None
                        audio_key = content["audio"]
                        audio_media = data_dict["media"][audio_key]
                        # Extracted audio belongs to its own video media entry. Standalone
                        # waveforms retain the existing positional pairing convention.
                        paired_video_key = (
                            audio_key
                            if message_has_video and isinstance(audio_media, dict) and "videos" in audio_media
                            else active_video_key
                        )
                        active_video_timestamps = (
                            video_timestamps_by_key.get(paired_video_key) if paired_video_key is not None else None
                        )
                        if active_video_timestamps is None and message_has_video:
                            log.critical(
                                "[TokenizerDataError]paired audio must follow its video in the same user message "
                                f"so both modalities share one timestamp clock. url: {url}",
                                rank0_only=False,
                            )
                            return None
                        num_audio_tokens = int(audio_outputs["audio_token_lengths"][audio_index])
                        assert self.audio_processor is not None
                        audio_token_timestamps = self.audio_processor.get_token_timestamps(
                            int(audio_outputs["audio_feature_lengths"][audio_index])
                        )
                        # Cropped waveforms begin at zero locally; compare them to video on the source clock.
                        if active_video_timestamps is not None:
                            audio_token_timestamps = [
                                timestamp + audio_start_seconds[audio_index] for timestamp in audio_token_timestamps
                            ]
                        if self.audio_layout == "interleaved_av" and message_has_video:
                            previous_content_type = (
                                message["content"][content_idx - 1]["type"] if content_idx > 0 else None
                            )
                            if previous_content_type != "video" or paired_video_key != active_video_key:
                                log.critical(
                                    "[TokenizerDataError]interleaved_av requires matching adjacent [video, audio] "
                                    f"content pairs. url: {url}",
                                    rank0_only=False,
                                )
                                return None
                            assert active_video_timestamps is not None
                            audio_segment_lengths_by_video[-1] = get_audio_segment_token_lengths(
                                num_audio_tokens,
                                active_video_timestamps,
                                audio_token_timestamps=audio_token_timestamps,
                            )
                            audio_index += 1
                            continue

                        audio_start_token, audio_pad_token, audio_end_token = self.audio_special_tokens.tokens
                        if self.audio_layout == "separate_no_timestamps":
                            audio_timestamps: list[float] = []
                        else:
                            audio_timestamps = (
                                active_video_timestamps
                                if active_video_timestamps is not None
                                else get_audio_only_timestamps(
                                    num_audio_tokens=num_audio_tokens,
                                    temporal_patch_size=self.processor.temporal_patch_size,
                                    fps=self.audio_timestamp_fps,
                                    audio_token_timestamps=audio_token_timestamps,
                                )
                            )
                        content = {
                            "type": "text",
                            "text": expand_audio_placeholders_in_text(
                                audio_pad_token,
                                audio_outputs["audio_token_lengths"][audio_index : audio_index + 1],
                                audio_timestamps=[audio_timestamps],
                                audio_token_timestamps=[audio_token_timestamps],
                                audio_start_token=audio_start_token,
                                audio_pad_token=audio_pad_token,
                                audio_end_token=audio_end_token,
                            ),
                        }
                        audio_index += 1
                    new_content_list.append(content)
                message["content"] = new_content_list

        if audio_index != total_audios:
            raise RuntimeError(f"Processed {audio_index} audio clips, expected {total_audios}")

        if len(raw_images) > 0:
            data_dict["raw_image"] = raw_images  # each: [3,1,H,W]

        if len(raw_videos) > 0:
            data_dict["raw_video"] = raw_videos  # each: [3,T,H,W]

        # Retain the tool menu even when source system prose is stripped.
        if conversation and conversation[0].get("tools") is not None:
            processor_kwargs["tools"] = conversation[0]["tools"]

        if self.strip_original_system_prompt:
            conversation = [msg for msg in conversation if msg.get("role") != "system"]
            data_dict["conversation"] = conversation

        if self.custom_system_prompt is not None and conversation[0]["role"] != "system":
            conversation.insert(0, {"role": "system", "content": self.custom_system_prompt})

        if self.text_only and (total_images > 0 or total_videos > 0 or total_audios > 0):
            log.critical(
                f"Images, videos, or audios found in the conversation but expect only text, __url__: {url} | "
                f"data_dict: {data_dict.keys()} | conversation={conversation}"
            )
            return None

        add_vision_id = self.add_vision_id and (total_images > 1 or total_videos > 1)

        try:
            conversation = convert_all_images_to_rgb(conversation)
        except Exception as e:
            if self.video_timestamp_mode == "source_pts":
                raise ValueError("source_pts failed while converting selected video frames to RGB") from e
            log.critical(
                f"Error in convert_all_images_to_rgb: {e} | conversation: {conversation} | __url__: {url} | data_dict: {data_dict.keys()}"
            )
            return None

        def tokenize_conversation() -> dict[str, torch.Tensor]:
            tokenizer_output = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=add_vision_id,
                **processor_kwargs,
            )
            if self.audio_layout == "interleaved_av" and any(
                segment_lengths is not None for segment_lengths in audio_segment_lengths_by_video
            ):
                assert self.audio_special_tokens is not None
                spliced_input_ids = splice_audio_segments_after_video_chunks(
                    tokenizer_output["input_ids"],
                    tokenizer_output["video_grid_thw"][:, 0],
                    audio_segment_lengths_by_video,
                    audio_token_ids=self.audio_special_tokens.token_ids,
                    video_pad_token_id=cast(int, self.processor.video_token_id),
                    vision_end_token_id=cast(
                        int,
                        self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
                    ),
                )
                tokenizer_output["input_ids"] = spliced_input_ids
                tokenizer_output["attention_mask"] = tokenizer_output["attention_mask"].new_ones(
                    spliced_input_ids.shape
                )
            return tokenizer_output

        adaptive_video_budget_applied = False
        try:
            tokenizer_output = tokenize_conversation()
            if self.adaptive_video_token_budget and total_videos > 0:
                assert self.max_total_token_length is not None
                initial_video_token_counts = self._video_token_counts(tokenizer_output)
                if len(initial_video_token_counts) != total_videos:
                    raise ValueError(f"Expected {total_videos} video grids, got {len(initial_video_token_counts)}.")
                minimum_retained_video_tokens = tuple(
                    min(self.min_video_token_length, token_count) for token_count in initial_video_token_counts
                )
                minimum_video_tokens = sum(minimum_retained_video_tokens)
                current_video_budget = self._count_video_tokens(tokenizer_output)
                for _ in range(3):
                    total_token_length = int(tokenizer_output["input_ids"].numel())
                    if total_token_length <= self.max_total_token_length:
                        break

                    actual_video_tokens = self._count_video_tokens(tokenizer_output)
                    non_video_tokens = total_token_length - actual_video_tokens
                    hard_available_video_tokens = self.max_total_token_length - non_video_tokens
                    available_video_tokens = (
                        self.max_total_token_length - self.adaptive_video_token_safety_margin - non_video_tokens
                    )
                    if hard_available_video_tokens < minimum_video_tokens:
                        log.critical(
                            "[TokenizerDataError]text and non-video tokens leave too little room for video: "
                            f"total={total_token_length}, non_video={non_video_tokens}, "
                            f"hard_available_video={hard_available_video_tokens}, "
                            f"minimum_video={minimum_video_tokens}, max_total={self.max_total_token_length}. "
                            f"url: {url}",
                            rank0_only=False,
                        )
                        return None
                    preferred_video_budget = max(
                        available_video_tokens,
                        minimum_video_tokens + self.adaptive_video_token_safety_margin,
                    )
                    next_video_budget = min(
                        self.max_video_token_length,
                        hard_available_video_tokens,
                        preferred_video_budget,
                        actual_video_tokens - 1,
                    )
                    if next_video_budget >= actual_video_tokens:
                        overflow = total_token_length - self.max_total_token_length
                        next_video_budget = actual_video_tokens - max(
                            1,
                            overflow + self.adaptive_video_token_safety_margin,
                        )
                    if next_video_budget < minimum_video_tokens:
                        log.critical(
                            "[TokenizerDataError]text and non-video tokens leave too little room for video: "
                            f"total={total_token_length}, non_video={non_video_tokens}, "
                            f"available_video={available_video_tokens}, minimum_video={minimum_video_tokens}, "
                            f"max_total={self.max_total_token_length}. url: {url}",
                            rank0_only=False,
                        )
                        return None
                    if not self._apply_video_token_budget(
                        conversation,
                        next_video_budget,
                        allow_frame_subsampling=total_audios == 0,
                    ):
                        log.critical(
                            "[TokenizerDataError]adaptive video budget is below the processor's minimum "
                            f"frame geometry: budget={next_video_budget}. url: {url}",
                            rank0_only=False,
                        )
                        return None
                    adaptive_video_budget_applied = True
                    current_video_budget = next_video_budget
                    tokenizer_output = tokenize_conversation()
                else:
                    total_token_length = int(tokenizer_output["input_ids"].numel())
                    if total_token_length > self.max_total_token_length:
                        log.critical(
                            "[TokenizerDataError]adaptive video tokenization did not fit after 3 retries: "
                            f"total={total_token_length}, max_total={self.max_total_token_length}, "
                            f"last_video_budget={current_video_budget}. url: {url}",
                            rank0_only=False,
                        )
                        return None

                if int(tokenizer_output["input_ids"].numel()) > self.max_total_token_length:
                    log.critical(
                        "[TokenizerDataError]adaptive video tokenization exceeded the exact sequence limit: "
                        f"total={int(tokenizer_output['input_ids'].numel())}, "
                        f"max_total={self.max_total_token_length}. url: {url}",
                        rank0_only=False,
                    )
                    return None
                if adaptive_video_budget_applied:
                    final_video_token_counts = self._video_token_counts(tokenizer_output)
                    if len(final_video_token_counts) != len(minimum_retained_video_tokens) or any(
                        actual < minimum
                        for actual, minimum in zip(
                            final_video_token_counts,
                            minimum_retained_video_tokens,
                            strict=True,
                        )
                    ):
                        log.critical(
                            "[TokenizerDataError]adaptive resize fell below the per-video quality floor after "
                            "patch-grid quantization: "
                            f"actual={final_video_token_counts}, minimum={minimum_retained_video_tokens}. "
                            f"url: {url}",
                            rank0_only=False,
                        )
                        return None
        except Exception as e:
            if self.video_timestamp_mode == "source_pts":
                raise ValueError("source_pts failed while rendering aligned video timestamps") from e
            log.critical(
                f"Error in tokenizer_output: {e} | conversation: {conversation} | __url__: {url} | data_dict: {data_dict.keys()}"
            )
            return None

        if adaptive_video_budget_applied and self.keep_raw_media:
            raw_videos = []
            for message in conversation:
                if not isinstance(message.get("content"), list):
                    continue
                for content in message["content"]:
                    if content.get("type") != "video":
                        continue
                    raw_video_frames = np.stack(
                        [np.asarray(frame.convert("RGB")) for frame in content["video"]],
                        axis=0,
                    )  # [T,H,W,3]
                    raw_videos.append(torch.from_numpy(raw_video_frames).permute(3, 0, 1, 2))  # [3,T,H,W]
            data_dict["raw_video"] = raw_videos
        input_ids = tokenizer_output["input_ids"]
        if audio_outputs is not None:
            assert self.audio_special_tokens is not None
            audio_pad_token_id = self.audio_special_tokens.token_ids[1]
            actual_audio_tokens = int((input_ids == audio_pad_token_id).sum())
            expected_audio_tokens = int(audio_outputs["audio_token_lengths"].sum())
            if actual_audio_tokens != expected_audio_tokens:
                log.critical(
                    "[TokenizerDataError]tokenized audio placeholder count does not match processor output: "
                    f"tokens={actual_audio_tokens}, expected={expected_audio_tokens}. url: {url}",
                    rank0_only=False,
                )
                return None
        if "image_grid_thw" in tokenizer_output and "raw_image" in data_dict:
            patch_size = self.processor.patch_size
            resized_raw_images: list[torch.Tensor] = []
            for raw_image, image_grid_thw in zip(data_dict["raw_image"], tokenizer_output["image_grid_thw"]):
                # image_grid_thw: [t,h,w] in patches
                _, h, w = image_grid_thw
                raw_image = torch.nn.functional.interpolate(
                    raw_image, size=(int(h) * patch_size, int(w) * patch_size), mode="bilinear", align_corners=False
                )  # [3,1,h*P,w*P]
                resized_raw_images.append(raw_image)
            data_dict["raw_image"] = resized_raw_images  # each: [3,1,h*P,w*P]

        try:
            # token_mask: True for tokens to compute loss on; False for tokens to ignore
            token_mask = self.processor.add_assistant_tokens_mask(input_ids)
            if hasattr(self.processor, "apply_supervision_overrides"):
                token_mask = self.processor.apply_supervision_overrides(input_ids, token_mask, conversation)
        except Exception as e:
            log.critical(
                f"Error in add_assistant_tokens_mask: {e} | conversation: {conversation} | __url__: {url} | data_dict: {data_dict.keys()}"
            )
            return None

        input_ids = torch.LongTensor(input_ids)  # [N_token]
        token_mask = torch.BoolTensor(token_mask)  # [N_token]; True = compute loss on this token

        data_dict.update({"input_ids": input_ids, "token_mask": token_mask})
        if data_stats is not None:
            data_dict["data_stats"] = data_stats.finalize(
                input_ids=input_ids,
                processor=self.processor,
                system_prompt=system_prompt_from(conversation),
                is_thinking_stripped=bool(data_dict.get("is_thinking_stripped", False)),
                max_image_token_length=self.max_image_token_length,
                max_video_token_length=self.max_video_token_length,
            )
        if audio_outputs is not None:
            data_dict.update(audio_outputs)
        for key in PROCESSOR_KEYS_TO_ADD:
            if key in tokenizer_output:
                data_dict[key] = tokenizer_output[key]
        if (
            "mm_token_type_ids" in data_dict
            and data_dict["mm_token_type_ids"].shape != input_ids.shape
            and hasattr(self.processor, "build_mm_token_type_ids")
        ):
            log.warning(
                "Tokenized mm_token_type_ids with shape "
                f"{data_dict['mm_token_type_ids'].shape} do not align with input_ids shape {input_ids.shape}; "
                "rebuilding modality IDs"
            )
            data_dict["mm_token_type_ids"] = self.processor.build_mm_token_type_ids(input_ids)
        labels = tokenizer_output["input_ids"].clone()  # [N_token]
        labels[~token_mask] = IGNORE_INDEX
        data_dict["labels"] = labels
        data_dict["pad_token_id"] = self.processor.pad_id
        data_dict["ignore_index"] = IGNORE_INDEX

        # keep raw text for debugging/logging purpose. Add \n\n after each <|im_end|>.
        dialog_str = self.processor.decode(input_ids)
        data_dict["dialog_str"] = compress_repeated_tokens(dialog_str.replace("<|im_end|>", "<|im_end|>\n\n"))

        # For debugging purpose
        msg = f"input_ids: {input_ids.shape[-1]} | __url__: {data_dict['__url__'].root}, {data_dict['__url__'].path} | __key__: {data_dict['__key__']}"
        if "raw_video" in data_dict:
            raw_video = data_dict["raw_video"]
            if isinstance(raw_video, list):
                msg += f" | raw_video: {[video.shape for video in raw_video]} "
            else:
                msg += f" | raw_video: {raw_video.shape} "
        if "raw_image" in data_dict:
            raw_image = data_dict["raw_image"]
            if isinstance(raw_image, list):
                msg += f" | raw_image: {[image.shape for image in raw_image]} "
            else:
                msg += f" | raw_image: {raw_image.shape} "
        if "pixel_values" in data_dict:
            msg += f" | pixel_values: {data_dict['pixel_values'].shape} "

        msg += f"original conversation: {data_dict['conversation']}"

        return data_dict
