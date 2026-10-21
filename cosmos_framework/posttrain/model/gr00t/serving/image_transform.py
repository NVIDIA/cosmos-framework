# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The GR00T eval image transform, matching the pipeline the checkpoint was TRAINED with.

vLLM-Omni's GR00T port reimplements Isaac-GR00T's eval image transform in torchvision and does not
reproduce it: it crops to the fixed ``image_crop_size`` instead of ``crop_fraction`` of the resized
image (230 vs 243 px for the DROID checkpoint), resizes bilinearly instead of ``cv2.INTER_AREA``, and
letterboxes unconditionally instead of on ``letter_box_transform``. The checkpoint was trained through
Isaac's pipeline, so Isaac is correct; the parity check traced every downstream divergence to this.

No Isaac-GR00T import: this is rollout code and Isaac-GR00T is a training-side dependency. The
rounding rules are transcribed from albumentations 1.4.18 ``_func_max_size`` and Isaac's
``FractionalCenterCrop``; ``tests/test_roles_rollout_gr00t_image_transform.py`` pins the result
against the real Isaac transform bit for bit.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

import numpy as np
import torch

PROCESSOR_CONFIG = "processor_config.json"


def _smallest_edge_resize(image: np.ndarray, max_size: int) -> np.ndarray:
    """albumentations ``SmallestMaxSize(max_size, INTER_AREA)``.

    Scale off the SHORTEST edge, skip entirely at scale 1.0, round each dimension independently;
    ``cv2.resize`` takes ``dsize`` as ``(width, height)``.
    """
    import cv2

    height, width = image.shape[:2]
    scale = max_size / float(min(height, width))
    if scale == 1.0:
        return image
    new_height, new_width = (round(dim * scale) for dim in (height, width))
    if (new_height, new_width) == (height, width):
        return image
    return cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_AREA)


def _fractional_center_crop(image: np.ndarray, fraction: float) -> np.ndarray:
    """Isaac's ``FractionalCenterCrop``: ``int()`` truncation, not rounding, floor 1, centred."""
    height, width = image.shape[:2]
    crop_height = max(1, int(height * fraction))
    crop_width = max(1, int(width * fraction))
    y_min = (height - crop_height) // 2
    x_min = (width - crop_width) // 2
    return image[y_min : y_min + crop_height, x_min : x_min + crop_width]


def _letter_box_pad(image: np.ndarray) -> np.ndarray:
    """Isaac's ``LetterBoxPad``: zero-pad to square, centred, remainder on the far side."""
    import cv2

    height, width = image.shape[:2]
    if height == width:
        return image
    longest = max(height, width)
    pad_h, pad_w = longest - height, longest - width
    top, left = pad_h // 2, pad_w // 2
    return cv2.copyMakeBorder(image, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=0)


@dataclasses.dataclass(frozen=True)
class Gr00tEvalImageTransform:
    """Per-frame eval transform: HWC uint8 ndarray in, CHW uint8 tensor out.

    That is the port's contract at its single call site, ``torch.stack([image_transform(img) ...])``
    followed by a uint8/3-channel assertion, so installing this replaces exactly one attribute.
    """

    image_target_size: tuple[int, int]
    image_crop_size: tuple[int, int]
    shortest_image_edge: int | None
    crop_fraction: float | None
    letter_box_transform: bool
    use_albumentations: bool

    @property
    def name(self) -> str:
        """Branch identifier, surfaced through ``describe_pipeline`` so the install is observable."""
        return "reference-albumentations" if self.use_albumentations else "reference-torchvision"

    @property
    def fraction(self) -> float:
        """Isaac's ``fraction_to_use``: explicit ``crop_fraction``, else the crop/target ratio."""
        if self.crop_fraction is not None:
            return float(self.crop_fraction)
        return self.image_crop_size[0] / self.image_target_size[0]

    @property
    def max_size(self) -> int:
        """Isaac's ``max_size``: explicit ``shortest_image_edge``, else the target's first extent."""
        edge = self.shortest_image_edge if self.shortest_image_edge is not None else self.image_target_size[0]
        return int(edge)

    def __call__(self, image: Any) -> torch.Tensor:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                f"GR00T eval image transform expects HWC uint8 RGB, got shape={array.shape} dtype={array.dtype}"
            )
        out = self._albumentations(array) if self.use_albumentations else self._torchvision(array)
        # A crop is a non-contiguous view; the port stacks these into one tensor immediately after.
        return torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1)

    def _albumentations(self, image: np.ndarray) -> np.ndarray:
        """resize -> fractional centre crop -> resize, on the shortest edge with INTER_AREA."""
        if self.letter_box_transform:
            image = _letter_box_pad(image)
        image = _smallest_edge_resize(image, self.max_size)
        image = _fractional_center_crop(image, self.fraction)
        return _smallest_edge_resize(image, self.max_size)

    def _torchvision(self, image: np.ndarray) -> np.ndarray:
        """Isaac's non-albumentations branch, for a checkpoint built that way."""
        from torchvision.transforms import v2 as transforms

        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        steps: list[Any] = []
        if self.letter_box_transform:
            steps.append(_LetterBoxTensor())
        steps += [
            transforms.Resize(size=list(self.image_target_size)),
            transforms.CenterCrop(size=list(self.image_crop_size)),
            transforms.Resize(size=list(self.image_target_size)),
        ]
        for step in steps:
            tensor = step(tensor)
        return tensor.permute(1, 2, 0).numpy()

    @classmethod
    def from_processor(cls, processor: Any, *, model_path: str) -> Gr00tEvalImageTransform:
        """Build from the CHECKPOINT's ``processor_config.json``, never from defaults.

        ``use_albumentations`` must come from the file: the port swallows it into ``**kwargs`` and
        discards it, so it is not an attribute of the loaded processor. It selects between two pipelines
        that produce different pixels, so its absence raises rather than defaulting.
        """
        kwargs = _processor_kwargs(model_path)
        if "use_albumentations" not in kwargs:
            raise KeyError(
                f"{PROCESSOR_CONFIG} for {model_path!r} has no processor_kwargs.use_albumentations. "
                "It selects between two eval image pipelines that produce DIFFERENT pixels, so there "
                "is no safe default; record the checkpoint's intended value instead."
            )

        def pick(key: str, default: Any = None) -> Any:
            # The file is authoritative; the loaded processor is the fallback for keys it does keep.
            return kwargs.get(key, getattr(processor, key, default))

        target, crop = pick("image_target_size"), pick("image_crop_size")
        if not target or not crop:
            raise KeyError(f"{PROCESSOR_CONFIG} for {model_path!r} lacks image_target_size/image_crop_size")
        return cls(
            image_target_size=(int(target[0]), int(target[1])),
            image_crop_size=(int(crop[0]), int(crop[1])),
            shortest_image_edge=pick("shortest_image_edge"),
            crop_fraction=pick("crop_fraction"),
            letter_box_transform=bool(pick("letter_box_transform", False)),
            use_albumentations=bool(kwargs["use_albumentations"]),
        )


class _LetterBoxTensor:
    """Zero-pad a CHW tensor to square, matching Isaac's torchvision ``LetterBoxTransform``."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms import v2 as transforms

        height, width = tensor.shape[-2:]
        if height == width:
            return tensor
        longest = max(height, width)
        pad_h, pad_w = longest - height, longest - width
        top, left = pad_h // 2, pad_w // 2
        return transforms.functional.pad(tensor, padding=[left, top, pad_w - left, pad_h - top], fill=0)


def _processor_kwargs(model_path: str) -> dict[str, Any]:
    """Read ``processor_config.json``'s ``processor_kwargs`` from a local dir or an HF repo id."""
    local = os.path.join(model_path, PROCESSOR_CONFIG)
    if os.path.isfile(local):
        path = local
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=model_path, filename=PROCESSOR_CONFIG)
    with open(path) as handle:
        config = json.load(handle)
    kwargs = config.get("processor_kwargs")
    if not isinstance(kwargs, dict):
        raise KeyError(f"{path} has no processor_kwargs mapping")
    return kwargs


def install_reference_image_transform(policy: Any, model_path: str) -> Gr00tEvalImageTransform:
    """Replace a loaded ``Gr00tPolicy``'s eval image transform with the reference one.

    One attribute: the port calls ``self.eval_image_transform`` per frame and does nothing else
    image-specific. Used by both the RL pipeline and the parity script's in-process side, so the two
    cannot drift.
    """
    transform = Gr00tEvalImageTransform.from_processor(policy.processor, model_path=model_path)
    policy.processor.eval_image_transform = transform
    return transform
