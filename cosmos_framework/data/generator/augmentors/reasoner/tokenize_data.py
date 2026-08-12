# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Visual-Text Transformations or Augmentations."""

import re
from typing import Any, Dict, Optional, Protocol, cast

import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils import log
from cosmos_framework.data.generator.reasoner.video_decoder_qwen import token_to_pixels
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
from cosmos_framework.data.generator.processors.qwen3vl_processor import Qwen3VLProcessor as Processor
from cosmos_framework.utils.generator.reasoner.constant import IGNORE_INDEX, PROCESSOR_KEYS_TO_ADD


class _AudioProcessor(Protocol):
    sampling_rate: int

    def __call__(
        self,
        audios: Any,
        *,
        sampling_rate: int,
    ) -> dict[str, torch.Tensor]: ...

    def get_token_timestamps(self, audio_feature_length: int) -> list[float]: ...


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
        max_video_token_length: int = 8192,
        max_image_token_length: int = 8192,
        custom_system_prompt: str | None = None,
        strip_original_system_prompt: bool = False,
        text_only: bool = False,
        sound_und: bool = False,
        audio_processor: Optional[_AudioProcessor] = None,
        audio_encoder_type: str = "parakeet",
        audio_start_token: str = AUDIO_START_TOKEN,
        audio_pad_token: str = AUDIO_PAD_TOKEN,
        audio_end_token: str = AUDIO_END_TOKEN,
        audio_timestamp_fps: float = DEFAULT_REASONER_VIDEO_FPS,
        audio_layout: str = "separate_with_timestamps",
    ) -> None:
        """
        Args:
            processor (Processor): Text/Image processor for tokenization.
            max_video_token_length (int): Maximum number of video tokens to use. Defaults to 8192.
            custom_system_prompt: Prompt to inject when no leading system message exists.
            strip_original_system_prompt: Remove existing system messages before optional injection.
            sound_und (bool): Opt in to audio preprocessing and audio-token registration.
                Disabled by default so existing text/vision tokenizers are unchanged.
        """
        # Create the tokenizer
        self.text_only = text_only
        self.processor = processor  # Expecting a ImageTextTokenizer
        self.max_video_token_length = max_video_token_length
        self.max_image_token_length = max_image_token_length
        self.custom_system_prompt = custom_system_prompt
        self.strip_original_system_prompt = strip_original_system_prompt
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
        processor_kwargs = {}
        total_images = 0
        total_videos = 0
        total_audios = 0
        raw_images: list[torch.Tensor] = []
        raw_videos: list[torch.Tensor] = []
        audio_clips: list[np.ndarray | torch.Tensor] = []
        # Pre-compute modality counts. Audio clips follow the same typed-content
        # -> media-dictionary schema as images and videos.
        for message in conversation:
            if not isinstance(message, dict):
                raise ValueError(
                    f"message is not a dict: {message} | conversation: {conversation} | data_dict: {data_dict} | __url__: {data_dict['__url__'].root}, {data_dict['__url__'].path}"
                )
            if message["role"] == "user" and isinstance(message["content"], list):
                total_images += len([content for content in message["content"] if content["type"] == "image"])
                total_videos += len([content for content in message["content"] if content["type"] == "video"])
                total_audios += len([content for content in message["content"] if content["type"] == "audio"])

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
                if message["role"] != "user" or not isinstance(message["content"], list):
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
            active_video_timestamps: list[float] | None = None
            # for user message, we insert the media
            if message["role"] == "user" and isinstance(
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
                                    f"max_pixels_per_image: {max_pixels_per_image} < min_pixels_per_image: {min_pixels_per_image} | self.max_video_token_length = {self.max_video_token_length} is not enough for total_images: {total_images}, as the default min_pixels is {min_pixels_per_image} | Either increase max_video_token_length or include max_pixels in the content or reduce min_pixels"
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
                        raw_image = np.asarray(image.convert("RGB"))  # [H,W,3]
                        raw_images.append(torch.from_numpy(raw_image).permute(2, 0, 1)[:, None])  # [3,1,H,W]

                    elif content["type"] == "video":
                        # as tokenization will NOT upsample the video, we can use a larger value here at the cost of multiple video having 1.5x token length
                        max_total_pixels = token_to_pixels(self.max_video_token_length * 1.5, temporal_patch_size=2)
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
                        if "videos" not in data_dict["media"][media_key]:
                            log.info(
                                f"[TokenizerDataError]videos not found in media[{media_key}], available keys: {data_dict['media'][media_key].keys()}. url: {url}"
                            )
                            return None
                        videos = data_dict["media"][media_key]["videos"]  # list of PIL images
                        fps = data_dict["media"][media_key]["fps"]
                        # this is because videos are decoded to be around "max_video_token_length" tokens

                        videos = maybe_subsample_frames(
                            self.processor.name, videos, self.max_video_token_length, self.processor
                        )
                        if len(videos) == 0:
                            log.info(f"[TokenizerDataError]video {media_key} has no decoded frames. url: {url}")
                            return None
                        content["video"] = videos

                        max_pixels_per_image = max_total_pixels // total_videos // len(videos)
                        content["fps"] = fps
                        content["max_pixels"] = max_pixels_per_image
                        if message_has_audio:
                            active_video_timestamps = get_qwen_video_timestamps(
                                num_frames=len(videos),
                                fps=fps,
                                temporal_patch_size=self.processor.temporal_patch_size,
                            )
                        if self.audio_layout == "interleaved_av":
                            audio_segment_lengths_by_video.append(None)

                        raw_video_frames = np.stack(
                            [np.asarray(frame.convert("RGB")) for frame in videos], axis=0
                        )  # [T,H,W,3]
                        raw_videos.append(torch.from_numpy(raw_video_frames).permute(3, 0, 1, 2))  # [3,T,H,W]
                    elif content["type"] == "audio":
                        assert audio_outputs is not None
                        assert self.audio_special_tokens is not None
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
                        if self.audio_layout == "interleaved_av" and message_has_video:
                            previous_content_type = (
                                message["content"][content_idx - 1]["type"] if content_idx > 0 else None
                            )
                            if previous_content_type != "video":
                                log.critical(
                                    "[TokenizerDataError]interleaved_av requires adjacent [video, audio] "
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

        if total_images > 1 or total_videos > 1:
            add_vision_id = True
        else:
            add_vision_id = False

        try:
            conversation = convert_all_images_to_rgb(conversation)
        except Exception as e:
            log.critical(
                f"Error in convert_all_images_to_rgb: {e} | conversation: {conversation} | __url__: {url} | data_dict: {data_dict.keys()}"
            )
            return None

        try:
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
        except Exception as e:
            log.critical(
                f"Error in tokenizer_output: {e} | conversation: {conversation} | __url__: {url} | data_dict: {data_dict.keys()}"
            )
            return None
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
            resized_raw_images: list[torch.Tensor] = []
            for raw_image, image_grid_thw in zip(data_dict["raw_image"], tokenizer_output["image_grid_thw"]):
                # image_grid_thw: [t,h,w]
                _, h, w = image_grid_thw
                raw_image = torch.nn.functional.interpolate(
                    raw_image, size=(int(h) * 14, int(w) * 14), mode="bilinear", align_corners=False
                )  # [3,1,h*14,w*14]
                resized_raw_images.append(raw_image)
            data_dict["raw_image"] = resized_raw_images  # each: [3,1,h*14,w*14]

        try:
            # token_mask: True for tokens to compute loss on; False for tokens to ignore
            token_mask = self.processor.add_assistant_tokens_mask(input_ids)
        except Exception as e:
            log.critical(
                f"Error in add_assistant_tokens_mask: {e} | conversation: {conversation} | __url__: {url} | data_dict: {data_dict.keys()}"
            )
            return None

        input_ids = torch.LongTensor(input_ids)  # [N_token]
        token_mask = torch.BoolTensor(token_mask)  # [N_token]; True = compute loss on this token

        data_dict.update(
            {
                "input_ids": input_ids,
                "token_mask": token_mask,
            }
        )
        if audio_outputs is not None:
            data_dict.update(audio_outputs)
        for key in PROCESSOR_KEYS_TO_ADD:
            if key in tokenizer_output:
                data_dict[key] = tokenizer_output[key]
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
