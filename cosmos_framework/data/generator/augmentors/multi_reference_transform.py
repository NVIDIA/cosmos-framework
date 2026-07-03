# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Augmentors for multi-reference image generation.

Multi-reference data layout (per sample):
    annotations: dict with keys {instruction, raw_instruction, in_instruction,
        input_images, output_image, editing_type, dataset_name, split}
    images: dict mapping {input_01, input_02, ..., input_NN, output} -> raw JPEG bytes

The ``instruction`` field may be either a single string (legacy format) or a
dict of prompt variants ``{original, short, medium, detailed}`` (new format).
For the dict form, one variant is sampled uniformly at random per sample.

These augmentors transform that on-disk format into the same in-memory layout
expected by ``ImageEditingToTrainingFormat`` (i.e. ``source_image`` is a list of
PIL images, ``target_image`` is a single PIL image, ``editing_instruction`` is a
string), so that the downstream image-editing augmentors can be reused unchanged.
"""

from __future__ import annotations

import io
import random
import re
from typing import Optional

from PIL import Image, UnidentifiedImageError

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils import log

Image.MAX_IMAGE_PIXELS = 933120000

_INPUT_KEY_RE = re.compile(r"^input_(\d+)$")
_OUTPUT_KEY = "output"

# The ``instruction`` annotation field may be either a single string (legacy
# format) or a dict of prompt variants at different lengths (new format). When
# it is a dict, one variant is sampled uniformly at random per sample. These are
# the variant keys we sample over, in their canonical order.
_PROMPT_VARIANT_KEYS = ("original", "short", "medium", "detailed")


def _sorted_input_keys(keys: list[str]) -> list[str]:
    """Return ``input_NN`` keys sorted by their numeric index."""
    indexed: list[tuple[int, str]] = []
    for key in keys:
        match = _INPUT_KEY_RE.match(key)
        if match is not None:
            indexed.append((int(match.group(1)), key))
    indexed.sort(key=lambda x: x[0])
    return [key for _, key in indexed]


class MultiReferencePKLToMedia(Augmentor):
    """Decode the multi-reference image bundle into PIL images.

    Reads ``data_dict[input_key]`` (a ``dict[str, bytes]`` whose keys are
    ``input_01``..``input_NN`` plus ``output``) and writes the decoded PIL
    images into ``data_dict[output_key]`` as ``dict[str, PIL.Image]`` with the
    same keys.

    Returns ``None`` (skip sample) if any image fails to decode or if the
    ``output`` image is missing — both are required for training.
    """

    def __init__(
        self,
        input_keys: list | None = None,
        input_key: str = "images",
        output_key: str = "media_list",
        args: Optional[dict] = None,
    ) -> None:
        super().__init__(input_keys or [], args=args)
        self.input_key = input_key
        self.output_key = output_key

    def _bytes_to_pil(self, image_bytes: bytes, identifier: str) -> Image.Image | None:
        try:
            with io.BytesIO(image_bytes) as stream:
                img = Image.open(stream)
                img.load()
                return img.convert("RGB")
        except UnidentifiedImageError:
            log.warning(
                f"Skipping item '{identifier}': cannot identify image bytes.",
                rank0_only=False,
            )
        except Exception as e:
            log.warning(
                f"Skipping item '{identifier}': error decoding image bytes: {e}",
                rank0_only=False,
            )
        return None

    def __call__(self, data_dict: dict) -> dict | None:
        if self.input_key not in data_dict:
            log.warning(
                f"Input key '{self.input_key}' not found in data_dict (keys={list(data_dict.keys())})",
                rank0_only=False,
            )
            return None

        bundle = data_dict[self.input_key]
        if not isinstance(bundle, dict):
            log.warning(
                f"Expected dict at data_dict['{self.input_key}'], got {type(bundle)}",
                rank0_only=False,
            )
            return None

        media: dict[str, Image.Image] = {}
        for name, item in bundle.items():
            if not isinstance(item, (bytes, bytearray)):
                log.warning(
                    f"Skipping item '{name}': expected bytes, got {type(item)}",
                    rank0_only=False,
                )
                return None
            decoded = self._bytes_to_pil(bytes(item), identifier=f"{self.input_key}['{name}']")
            if decoded is None:
                return None
            media[name] = decoded

        if _OUTPUT_KEY not in media:
            log.warning(
                f"Multi-reference sample missing '{_OUTPUT_KEY}' image: {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None
        if not _sorted_input_keys(list(media.keys())):
            log.warning(
                f"Multi-reference sample has no 'input_NN' images: {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        data_dict[self.output_key] = media
        if self.input_key != self.output_key:
            del data_dict[self.input_key]

        return data_dict


class ExtractMultiReferenceConversation(Augmentor):
    """Extract source/target images and instruction from multi-reference annotation.

    Expected ``data_dict`` state on entry:
        annotations: dict with at least ``instruction`` and (implicitly via the
            image bundle) ``input_NN``/``output`` images. ``in_instruction`` is
            present but intentionally ignored — per pipeline design we keep all
            input images in the order indicated by their ``input_NN`` key.
        diffusion_media_list: dict[str, PIL.Image] keyed by the same
            ``input_NN``/``output`` names, populated by
            ``InterleavedMediaResize`` upstream.

    The ``instruction`` field supports two formats:
        - str: a single instruction (legacy format), used directly.
        - dict: prompt variants keyed by ``original``/``short``/``medium``/
          ``detailed`` (new format); one non-empty variant is sampled uniformly
          at random per sample.

    Output additions to ``data_dict``:
        source_image: list[PIL.Image] in input-index order, truncated to
            ``max_reference_images`` if necessary.
        target_image: PIL.Image (the ``output`` image).
        editing_instruction: str (the sampled ``instruction``, with
            ``<img-1>, <img-2>, ...`` markers preserved).
        dataset_name: ``f"{annotations.dataset_name}/{annotations.split}"``
            (used for logging only).

    Returns ``None`` when required fields are missing.
    """

    def __init__(
        self,
        input_keys: list | None = None,
        max_reference_images: int = 20,
        annotation_key: str = "annotations",
        media_key: str = "diffusion_media_list",
        instruction_key: str = "instruction",
        prompt_variant_keys: tuple[str, ...] | list[str] = _PROMPT_VARIANT_KEYS,
        args: Optional[dict] = None,
    ) -> None:
        super().__init__(input_keys or [], args=args)
        if max_reference_images <= 0:
            raise ValueError(f"max_reference_images must be positive, got {max_reference_images}")
        self.max_reference_images = max_reference_images
        self.annotation_key = annotation_key
        self.media_key = media_key
        self.instruction_key = instruction_key
        self.prompt_variant_keys = tuple(prompt_variant_keys)

    def _resolve_instruction(self, annotation: dict) -> str | None:
        """Resolve the instruction string, sampling a variant when given a dict.

        Returns a non-empty, stripped instruction string, or ``None`` when no
        usable instruction is present.
        """
        raw = annotation.get(self.instruction_key)

        if isinstance(raw, str):
            return raw.strip() or None

        if isinstance(raw, dict):
            # Prefer the canonical variant keys (in order), then fall back to any
            # other string values so we never silently drop a usable prompt.
            candidates = [
                raw[key].strip()
                for key in self.prompt_variant_keys
                if isinstance(raw.get(key), str) and raw[key].strip()
            ]
            if not candidates:
                candidates = [value.strip() for value in raw.values() if isinstance(value, str) and value.strip()]
            if candidates:
                return random.choice(candidates)

        return None

    def __call__(self, data_dict: dict) -> dict | None:
        for required_key in (self.annotation_key, self.media_key):
            if required_key not in data_dict:
                log.warning(
                    f"'{required_key}' not found in data_dict: {data_dict.get('__key__', 'unknown')}",
                    rank0_only=False,
                )
                return None

        annotation = data_dict[self.annotation_key]
        if not isinstance(annotation, dict):
            log.warning(
                f"Expected dict for '{self.annotation_key}', got {type(annotation)}: "
                f"{data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        instruction = self._resolve_instruction(annotation)
        if not instruction:
            log.warning(
                f"Missing/empty '{self.instruction_key}' in annotation: {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        media_dict = data_dict[self.media_key]
        if not isinstance(media_dict, dict):
            log.warning(
                f"Expected dict for '{self.media_key}', got {type(media_dict)}: {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        target_image = media_dict.get(_OUTPUT_KEY)
        if isinstance(target_image, list):
            target_image = target_image[0] if target_image else None
        if target_image is None:
            log.warning(
                f"Missing '{_OUTPUT_KEY}' image after resize: {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        ordered_input_keys = _sorted_input_keys(list(media_dict.keys()))
        if not ordered_input_keys:
            log.warning(
                f"No 'input_NN' images after resize: {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        if len(ordered_input_keys) > self.max_reference_images:
            ordered_input_keys = ordered_input_keys[: self.max_reference_images]

        source_images: list[Image.Image] = []
        for key in ordered_input_keys:
            ref = media_dict[key]
            if isinstance(ref, list):
                ref = ref[0] if ref else None
            if ref is None:
                log.warning(
                    f"Reference image '{key}' is None after resize: {data_dict.get('__key__', 'unknown')}",
                    rank0_only=False,
                )
                return None
            source_images.append(ref)

        data_dict["source_image"] = source_images
        data_dict["target_image"] = target_image
        data_dict["editing_instruction"] = instruction

        ds_name = annotation.get("dataset_name")
        split = annotation.get("split")
        if isinstance(ds_name, str) and ds_name:
            data_dict["dataset_name"] = f"{ds_name}/{split}" if isinstance(split, str) and split else ds_name

        return data_dict


class RandomResizeReferenceImages(Augmentor):
    """Randomly rescale each reference image (aspect ratio preserved); leave the target untouched.

    Operates on the ``media_list`` dict produced by :class:`MultiReferencePKLToMedia`,
    i.e. before :class:`InterleavedMediaResize`. With probability ``resize_prob`` (one
    Bernoulli draw per sample), every key matching ``input_NN`` gets its own independently
    sampled ratio in ``[min_resize_ratio, max_resize_ratio]`` and is resized via LANCZOS.
    The ``output`` key (target image) is skipped so the target resolution is unchanged.

    Padding-divisor alignment and the side-length cap are handled by the downstream
    :class:`InterleavedMediaResize` stage, so this augmentor does not need to worry
    about VAE divisibility.
    """

    def __init__(
        self,
        input_keys: list | None = None,
        min_resize_ratio: float = 0.5,
        max_resize_ratio: float = 1.5,
        resize_prob: float = 0.0,
        media_key: str = "media_list",
        output_key_pattern: str = _OUTPUT_KEY,
        args: Optional[dict] = None,
    ) -> None:
        if not 0.0 <= resize_prob <= 1.0:
            raise ValueError(f"resize_prob must be in [0, 1], got {resize_prob}")
        if min_resize_ratio <= 0:
            raise ValueError(f"min_resize_ratio must be > 0, got {min_resize_ratio}")
        if max_resize_ratio < min_resize_ratio:
            raise ValueError(f"max_resize_ratio ({max_resize_ratio}) must be >= min_resize_ratio ({min_resize_ratio})")
        super().__init__(input_keys or [media_key], args=args)
        self.min_resize_ratio = float(min_resize_ratio)
        self.max_resize_ratio = float(max_resize_ratio)
        self.resize_prob = float(resize_prob)
        self.media_key = media_key
        self.output_key_pattern = output_key_pattern

    def _resize_one(self, img: Image.Image) -> Image.Image:
        ratio = random.uniform(self.min_resize_ratio, self.max_resize_ratio)
        w, h = img.size
        new_w = max(1, round(w * ratio))
        new_h = max(1, round(h * ratio))
        if (new_w, new_h) == (w, h):
            return img
        return img.resize((new_w, new_h), Image.LANCZOS)

    def __call__(self, data_dict: dict) -> dict | None:
        if self.resize_prob <= 0.0 or random.random() >= self.resize_prob:
            return data_dict

        media_list = data_dict.get(self.media_key)
        if not isinstance(media_list, dict):
            return data_dict

        for key, media in media_list.items():
            if key == self.output_key_pattern or _INPUT_KEY_RE.match(key) is None:
                continue
            if isinstance(media, list):
                media_list[key] = [self._resize_one(frame) for frame in media]
            elif isinstance(media, Image.Image):
                media_list[key] = self._resize_one(media)

        return data_dict


_MARKER_RE: re.Pattern[str] = re.compile(r"<img-(\d+)>")


class ReorderReferenceImages(Augmentor):
    """Shuffle the reference list and rewrite ``<img-N>`` markers in the instruction.

    Operates on the ``source_image`` list and ``editing_instruction`` string produced by
    :class:`ExtractMultiReferenceConversation`. With probability ``shuffle_prob`` (one
    Bernoulli draw per sample), shuffles ``source_image`` and consistently renumbers the
    ``<img-N>`` markers inside ``editing_instruction`` so that the same physical image is
    still referenced after the shuffle.

    Shuffling is **only** applied when ``editing_instruction`` contains at least one
    ``<img-N>`` marker. Some samples refer to references by natural-language phrases
    (``"Image 1"``, ``"first image"``, etc.) that we cannot reliably renumber, so
    shuffling those would silently desync the instruction from the image order. Such
    samples are returned unchanged.

    The marker rewrite is done in a single ``re.sub`` pass to avoid cascading rewrites
    (e.g. ``<img-1>`` -> ``<img-2>`` -> ``<img-3>``). Markers whose index falls outside
    ``[1, len(source_image)]`` are left untouched.
    """

    def __init__(
        self,
        input_keys: list | None = None,
        shuffle_prob: float = 0.0,
        source_key: str = "source_image",
        instruction_key: str = "editing_instruction",
        args: Optional[dict] = None,
    ) -> None:
        if not 0.0 <= shuffle_prob <= 1.0:
            raise ValueError(f"shuffle_prob must be in [0, 1], got {shuffle_prob}")
        super().__init__(input_keys or [], args=args)
        self.shuffle_prob = float(shuffle_prob)
        self.source_key = source_key
        self.instruction_key = instruction_key

    def __call__(self, data_dict: dict) -> dict | None:
        if self.shuffle_prob <= 0.0 or random.random() >= self.shuffle_prob:
            return data_dict

        source_image = data_dict.get(self.source_key)
        instruction = data_dict.get(self.instruction_key)
        if not isinstance(source_image, list) or len(source_image) <= 1:
            return data_dict
        if not isinstance(instruction, str) or not instruction:
            return data_dict

        # Only shuffle when the instruction actually uses ``<img-N>`` markers.
        # Otherwise the references are by natural-language phrases (e.g. "Image 1",
        # "the first image") that we can't reliably renumber, so reordering would
        # silently desync the instruction from the image order.
        if _MARKER_RE.search(instruction) is None:
            return data_dict

        n = len(source_image)
        # Try a few times to get a non-identity permutation; otherwise accept identity.
        perm = list(range(n))
        for _ in range(5):
            candidate = random.sample(range(n), n)
            if candidate != list(range(n)):
                perm = candidate
                break

        if perm == list(range(n)):
            return data_dict

        inv = [0] * n
        for new_idx, old_idx in enumerate(perm):
            inv[old_idx] = new_idx

        def _remap(match: re.Match) -> str:
            old_idx = int(match.group(1)) - 1
            if 0 <= old_idx < n:
                return f"<img-{inv[old_idx] + 1}>"
            return match.group(0)

        data_dict[self.source_key] = [source_image[perm[i]] for i in range(n)]
        data_dict[self.instruction_key] = _MARKER_RE.sub(_remap, instruction)

        return data_dict
