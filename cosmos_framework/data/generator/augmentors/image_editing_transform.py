# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Augmentors for image editing tasks in the cosmos3 VFM pipeline.

These augmentors process conversation-format image editing data and produce
the output format expected by the main training pipeline:
    - images: List[torch.Tensor] (source + target images as a two-frame "video")
    - image_size: List[torch.Tensor]
    - ai_caption: List[str]
    - selected_caption_type: List[str]
    - fps: List[float]
    - num_frames: List[int]
    - dataset_name: str
    - sequence_plan: SequencePlan
"""

from __future__ import annotations

import json
import random

import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils import log


class ExtractImageEditingConversation(Augmentor):
    """Extract and validate image editing conversation from standard annotation format.

    This augmentor processes cosmos-interleaved conversation data for image editing:
    - Validates that the conversation has exactly one round (user + assistant)
    - User message must contain at least one image and text instruction
    - Assistant message must contain exactly one image (the edited result)
    - If multi-round conversation is found, only the first round is kept

    Input Format (from data_dict):
        - texts: Dict containing "content" with conversation data
        - mllm_media_list: Dict mapping image keys to PIL images (for understanding)
        - diffusion_media_list: Dict mapping image keys to PIL images (for diffusion/VAE)
        - optional structured instruction key: Dict, JSON string, or JSON bytes containing
          text_json.content and gemini_rewrite. When configured, gemini_rewrite is used as
          the training prompt and text_json.content is used only to recover image references.

    Output Format (added to data_dict):
        - source_image: PIL.Image (the input image for editing)
        - target_image: PIL.Image (the edited output image)
        - editing_instruction: str (the user's editing instruction)
    """

    def __init__(
        self,
        input_keys: list | None = None,
        max_round: int = 1,
        instruction_key: str = "texts",
        conversation_key: str = "texts",
        structured_instruction_field: str | None = None,
        args: dict | None = None,
    ) -> None:
        super().__init__(input_keys or [], None, args)
        self.max_round: int = max_round
        self.instruction_key: str = instruction_key
        self.conversation_key: str = conversation_key
        self.structured_instruction_field: str | None = structured_instruction_field

    def _decode_json_text(self, text: str, payload_name: str, sample_key: str) -> dict | None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            log.warning(
                f"Error decoding {payload_name} JSON: {sample_key}, {str(e)}",
                rank0_only=False,
            )
            return None

        if not isinstance(payload, dict):
            log.warning(
                f"Decoded {payload_name} is not a dict: {sample_key}, got {type(payload)}",
                rank0_only=False,
            )
            return None
        return payload

    def _decode_payload(self, payload: object, payload_name: str, sample_key: str) -> dict | None:
        if isinstance(payload, dict):
            return payload

        if isinstance(payload, str):
            return self._decode_json_text(payload, payload_name, sample_key)

        if isinstance(payload, (bytes, bytearray)):
            try:
                text = bytes(payload).decode("utf-8")
            except UnicodeDecodeError as e:
                log.warning(
                    f"Error decoding {payload_name} bytes as UTF-8: {sample_key}, {str(e)}",
                    rank0_only=False,
                )
                return None
            return self._decode_json_text(text, payload_name, sample_key)

        log.warning(
            f"Unsupported {payload_name} payload type: {sample_key}, got {type(payload)}",
            rank0_only=False,
        )
        return None

    def _get_instruction_payload(self, data_dict: dict, sample_key: str) -> dict | None:
        payload = data_dict.get(self.instruction_key)
        if payload is None:
            log.warning(
                f"{self.instruction_key} not found in data_dict: {sample_key}",
                rank0_only=False,
            )
            return None
        return self._decode_payload(payload, self.instruction_key, sample_key)

    def _get_conversation_payload(
        self,
        data_dict: dict,
        instruction_payload: dict,
        sample_key: str,
    ) -> dict | None:
        if self.conversation_key == self.instruction_key:
            return instruction_payload

        if self.conversation_key in data_dict:
            return self._decode_payload(data_dict[self.conversation_key], self.conversation_key, sample_key)

        nested_payload = instruction_payload.get(self.conversation_key)
        if nested_payload is None:
            log.warning(
                f"{self.conversation_key} not found in {self.instruction_key}: {sample_key}",
                rank0_only=False,
            )
            return None
        return self._decode_payload(nested_payload, f"{self.instruction_key}.{self.conversation_key}", sample_key)

    def _get_structured_instruction(self, instruction_payload: dict, sample_key: str) -> str | None:
        if self.structured_instruction_field is None:
            return None

        rewrite_error = instruction_payload.get("rewrite_error")
        if rewrite_error is not None:
            log.warning(
                f"Structured instruction rewrite_error is non-null: {sample_key}, {rewrite_error}",
                rank0_only=False,
            )
            return None

        structured_payload = instruction_payload.get(self.structured_instruction_field)
        if not isinstance(structured_payload, dict):
            log.warning(
                f"{self.structured_instruction_field} missing or not a dict: {sample_key}",
                rank0_only=False,
            )
            return None

        edit_type = structured_payload.get("edit_type")
        structured_instruction = structured_payload.get("structured_instruction")
        if not isinstance(edit_type, str) or not edit_type:
            log.warning(f"Structured instruction edit_type missing: {sample_key}", rank0_only=False)
            return None
        if not isinstance(structured_instruction, dict) or not structured_instruction:
            log.warning(f"Structured instruction body missing: {sample_key}", rank0_only=False)
            return None

        prompt = {
            "edit_type": edit_type,
            "structured_instruction": structured_instruction,
        }
        return json.dumps(prompt, ensure_ascii=False)

    def __call__(self, data_dict: dict) -> dict | None:
        """Extract image editing conversation.

        Args:
            data_dict: Input data dictionary.

        Returns:
            Updated data_dict with source_image, target_image, editing_instruction,
            or None if the data is invalid.
        """
        # Validate required keys
        sample_key = data_dict.get("__key__", "unknown")
        for required_key in ["diffusion_media_list", self.instruction_key]:
            if required_key not in data_dict:
                log.warning(
                    f"{required_key} not found in data_dict: {sample_key}",
                    rank0_only=False,
                )
                return None

        diffusion_media_list = data_dict["diffusion_media_list"]
        instruction_payload = self._get_instruction_payload(data_dict, sample_key)
        if instruction_payload is None:
            return None
        conversation_payload = self._get_conversation_payload(data_dict, instruction_payload, sample_key)
        if conversation_payload is None:
            return None
        conversation_content_key = f"{self.conversation_key}.content"

        # Get conversation content
        try:
            texts_content = conversation_payload.get("content")
            if texts_content is None:
                log.warning(
                    f"{conversation_content_key} is None: {sample_key}",
                    rank0_only=False,
                )
                return None

            # Handle case where content is a list of conversation options
            if isinstance(texts_content, list) and len(texts_content) > 0:
                if isinstance(texts_content[0], list):
                    # Multiple conversation options, randomly select one
                    selected_conversations = random.choice(texts_content)
                else:
                    selected_conversations = texts_content
            else:
                log.warning(
                    f"Unexpected {conversation_content_key} format: {sample_key}",
                    rank0_only=False,
                )
                return None
        except Exception as e:
            log.warning(
                f"Error accessing {conversation_content_key}: {sample_key}, {str(e)}",
                rank0_only=False,
            )
            return None

        # For image editing, we only keep the first round (user + assistant)
        # Trim to first round if multiple rounds exist
        if len(selected_conversations) > 2:
            log.warning(
                f"Multi-round conversation found ({len(selected_conversations)} messages), "
                f"keeping only first round: {sample_key}",
                rank0_only=False,
            )
            selected_conversations = selected_conversations[:2]

        if len(selected_conversations) < 2:
            log.warning(
                f"Expected at least 2 messages (user + assistant), got {len(selected_conversations)}: {sample_key}",
                rank0_only=False,
            )
            return None

        # Validate roles: first must be user, second must be assistant
        user_msg = selected_conversations[0]
        assistant_msg = selected_conversations[1]

        if user_msg.get("role") != "user":
            log.warning(
                f"First message role is not 'user': {sample_key}",
                rank0_only=False,
            )
            return None

        if assistant_msg.get("role") != "assistant":
            log.warning(
                f"Second message role is not 'assistant': {sample_key}",
                rank0_only=False,
            )
            return None

        # Extract user content: must have at least one image and one text
        user_content = user_msg.get("content", [])
        if isinstance(user_content, str):
            user_content = [{"type": "text", "text": user_content}]

        user_text_parts: list[str] = []
        user_image_key: str | None = None

        for item in user_content:
            if not isinstance(item, dict):
                continue
            content_type = item.get("type")
            if content_type == "text":
                user_text_parts.append(item.get("text", ""))
            elif content_type == "image":
                if user_image_key is None:
                    user_image_key = item.get("image")
                # If multiple user images, we only take the first one

        if user_image_key is None:
            log.warning(
                f"No image found in user message: {sample_key}",
                rank0_only=False,
            )
            return None

        if self.structured_instruction_field is None:
            editing_instruction = " ".join(user_text_parts).strip()
            if not editing_instruction:
                log.warning(
                    f"No text instruction found in user message: {sample_key}",
                    rank0_only=False,
                )
                return None
        else:
            editing_instruction = self._get_structured_instruction(instruction_payload, sample_key)
            if editing_instruction is None:
                return None

        # Extract assistant content: must have exactly one image
        assistant_content = assistant_msg.get("content", [])
        if isinstance(assistant_content, str):
            log.warning(
                f"Assistant content is text-only (no image): {sample_key}",
                rank0_only=False,
            )
            return None

        assistant_image_key: str | None = None
        for item in assistant_content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                assistant_image_key = item.get("image")
                break

        if assistant_image_key is None:
            log.warning(
                f"No image found in assistant message: {sample_key}",
                rank0_only=False,
            )
            return None

        # Validate images exist in media lists
        for media_key in [user_image_key, assistant_image_key]:
            if media_key not in diffusion_media_list:
                log.warning(
                    f"Image {media_key} not found in diffusion_media_list: {sample_key}",
                    rank0_only=False,
                )
                return None

        # Get PIL images
        source_image = diffusion_media_list[user_image_key]
        target_image = diffusion_media_list[assistant_image_key]

        # Handle video (list of frames) - use first frame
        if isinstance(source_image, list):
            source_image = source_image[0] if source_image else None
        if isinstance(target_image, list):
            target_image = target_image[0] if target_image else None

        if source_image is None or target_image is None:
            log.warning(
                f"Source or target image is None: {sample_key}",
                rank0_only=False,
            )
            return None

        data_dict["source_image"] = source_image
        data_dict["target_image"] = target_image
        data_dict["editing_instruction"] = editing_instruction

        return data_dict


class ImageEditingToTrainingFormat(Augmentor):
    """Convert extracted image editing data to the training-compatible format.

    This augmentor takes the source image, target image, and editing instruction
    and produces the output format expected by the main training pipeline.

    Images are assumed to have been already resized by an upstream augmentor
    (e.g. ``OmniInterleavedMediaResize``).  This augmentor only normalises the
    PIL images to tensors and assembles the remaining metadata fields.

    Supports both single-source image editing and multi-reference generation:
    ``source_image`` may be a single ``PIL.Image`` (one reference) or a
    ``list[PIL.Image]`` (N references). The output ``images`` always places the
    target last, so the resulting layout is ``[ref_1, ..., ref_N, target]`` with
    ``num_frames = N + 1``.

    Input (from data_dict):
        - source_image: PIL.Image | list[PIL.Image] (already resized by upstream augmentor)
        - target_image: PIL.Image (already resized by upstream augmentor)
        - editing_instruction: str

    Output (added to data_dict):
        - images: list[torch.Tensor]  — ``[ref_1, ..., ref_N, target]`` (each C,H,W)
        - ai_caption: str
        - selected_caption_type: str
        - fps: float
        - num_frames: int
        - sequence_plan: SequencePlan
    """

    def __init__(
        self,
        input_keys: list | None = None,
        mean: float = 0.5,
        std: float = 0.5,
        args: dict | None = None,
    ) -> None:
        super().__init__(input_keys or [], None, args)
        self.mean = mean
        self.std = std

    def _normalize_image(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor (C, H, W)."""
        tensor = transforms_F.to_tensor(image)
        tensor = transforms_F.normalize(tensor, mean=[self.mean] * 3, std=[self.std] * 3)
        return tensor

    def __call__(self, data_dict: dict) -> dict | None:
        """Convert image editing data to training format.

        Args:
            data_dict: Input data dictionary with source_image, target_image, editing_instruction.

        Returns:
            Updated data_dict with training-compatible fields, or None on error.
        """
        source_image = data_dict.get("source_image")
        target_image: Image.Image = data_dict.get("target_image")
        editing_instruction: str = data_dict.get("editing_instruction", "")

        if source_image is None or target_image is None:
            return None

        # Support both single-source editing (one PIL image) and multi-reference
        # generation (a list of PIL images). Normalise to a list of references.
        source_images = [source_image] if isinstance(source_image, Image.Image) else list(source_image)
        if not source_images:
            return None

        try:
            # Normalize PIL images to tensors (upstream augmentor already handled resizing).
            # Each image keeps its own spatial size; the model encodes them separately.
            # The target is placed last: [ref_1, ..., ref_N, target].
            images = [self._normalize_image(src) for src in source_images]  # each [C,H_s,W_s]
            images.append(self._normalize_image(target_image))  # [C,H_t,W_t]

            # Store as list of tensors for the batch collation.
            data_dict["images"] = images

            # Set text fields
            data_dict["ai_caption"] = editing_instruction
            data_dict["selected_caption_type"] = "editing_instruction"

            # Set metadata
            data_dict["fps"] = 30.0  # Same as standard image training
            data_dict["num_frames"] = len(images)  # N references + target
            data_dict["image_size"] = [
                torch.tensor(
                    [img.height, img.width, img.height, img.width],
                    dtype=torch.float,
                )  # [4]
                for img in (*source_images, target_image)
            ]
            # Set the dataset name if not already present
            if "dataset_name" not in data_dict:
                data_dict["dataset_name"] = "image_editing"

            # Build sequence plan for image editing.
            # The number of vision items per sample (e.g. 2 for source + target) is tracked
            # by GenerationDataClean.num_vision_items_per_sample (set in get_data_and_condition).
            # In pack_input_sequence, all items except the last are fully conditioned;
            # the last item uses condition_frame_indexes_vision ([] = fully generated).
            from cosmos_framework.data.generator.sequence_packing import SequencePlan

            data_dict["sequence_plan"] = SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=[],  # Target (last item) is fully generated
            )

        except Exception as e:
            log.warning(
                f"Error processing image editing data: {data_dict.get('__key__', 'unknown')}, {str(e)}",
                rank0_only=False,
            )
            return None

        return data_dict


class RemoveKeys(Augmentor):
    """Remove specified keys from the data dictionary.

    This is useful for cleaning up intermediate keys that are not needed
    downstream (e.g. raw PIL images, media lists) so that every remaining
    value is a tensor, number, dict, or list — as required by the dataloader
    collation.

    Args:
        input_keys: Keys to remove from ``data_dict``.
    """

    def __init__(
        self,
        input_keys: list | None = None,
        args: dict | None = None,
    ) -> None:
        super().__init__(input_keys or [], None, args)

    def __call__(self, data_dict: dict) -> dict:
        for key in self.input_keys:
            data_dict.pop(key, None)
        return data_dict
