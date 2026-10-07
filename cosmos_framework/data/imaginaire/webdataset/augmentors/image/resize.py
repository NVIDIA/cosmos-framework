# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
from typing import Optional

import omegaconf
import torch
import torchvision.transforms.functional as transforms_F

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.data.imaginaire.webdataset.augmentors.image.misc import (
    obtain_aspect_ratio,
    obtain_augmentation_size,
    obtain_image_size,
)
from cosmos_framework.utils import log


class AspectRatioFilterStats:
    """Worker-local, per-bucket acceptance counts without a warning for every filtered clip."""

    def __init__(self) -> None:
        self.pid: int = os.getpid()
        self.counts: dict[tuple[str, tuple[int, int]], tuple[int, int]] = {}

    def record(self, aspect_ratio: str, target_size: tuple[int, int], accepted: bool) -> None:
        pid = os.getpid()
        if pid != self.pid:
            self.pid = pid
            self.counts.clear()
        key = (aspect_ratio, target_size)
        checked, rejected = self.counts.get(key, (0, 0))
        checked += 1
        rejected += int(not accepted)
        self.counts[key] = (checked, rejected)
        if checked == 100 or checked % 1000 == 0:
            log.info(
                f"Transfer aspect filter (worker-local): pid={pid}, aspect={aspect_ratio}, "
                f"target={target_size}, checked={checked}, accepted={checked - rejected}, "
                f"rejected={rejected}, acceptance={(checked - rejected) / checked:.2%}",
                rank0_only=False,
            )


class ResizeToSize(Augmentor):
    """Resize to the selected bucket, optionally filtering excessive aspect-ratio distortion.

    ``max_aspect_ratio_distortion`` is an inclusive fractional limit on
    ``max(scale_x, scale_y) / min(scale_x, scale_y) - 1``, independent of target pixel count.
    Rejected samples return None before resizing so the dataset can skip them.
    """

    def __init__(self, input_keys: list, output_keys: list | None = None, args: dict | None = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.aspect_ratio_stats: AspectRatioFilterStats = AspectRatioFilterStats()

    def __call__(self, data_dict: dict) -> dict | None:
        assert self.args is not None, "Please specify args in augmentations"
        img_size = obtain_augmentation_size(data_dict, self.args)
        assert isinstance(img_size, (tuple, list, omegaconf.listconfig.ListConfig)), (
            f"Arg size in resize should be a width-height pair, get {type(img_size)}, {img_size}"
        )
        target_w, target_h = map(int, img_size)
        max_distortion = data_dict.get("_res_max_aspect_ratio_distortion", self.args.get("max_aspect_ratio_distortion"))
        if max_distortion is not None:
            source_w, source_h = obtain_image_size(data_dict, self.input_keys)
            # Cross products compare the two scale factors using decoded dimensions,
            # independently of the metadata bucket label or overall downsampling.
            width_product = target_w * source_h
            height_product = target_h * source_w
            scale_ratio = max(width_product, height_product) / min(width_product, height_product)
            # Compare ratios directly so an exactly-on-limit sample is not rejected
            # by rounding from subtracting 1 (e.g. 1.03 - 1 > 0.03).
            accepted = scale_ratio <= 1.0 + max_distortion
            if self.args.get("log_aspect_ratio_stats", True):
                self.aspect_ratio_stats.record(obtain_aspect_ratio(data_dict), (target_w, target_h), accepted)
            if not accepted:
                return None

        output_keys = self.output_keys if self.output_keys is not None else self.input_keys
        for inp_key, out_key in zip(self.input_keys, output_keys):
            data_dict[out_key] = transforms_F.resize(  # [...,target_h,target_w]
                data_dict[inp_key],
                size=(target_h, target_w),
                interpolation=transforms_F.InterpolationMode.BICUBIC,
                antialias=True,
            )
            if out_key != inp_key:
                del data_dict[inp_key]

        # No padding: the complete resized frame is valid for both encoding and loss.
        data_dict["image_size"] = torch.tensor([target_h, target_w, target_h, target_w], dtype=torch.float)  # [4]
        return data_dict


class ResizeSmallestSide(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs resizing to smaller side

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            out_size = obtain_augmentation_size(data_dict, self.args)
            assert isinstance(out_size, int), "Arg size in resize should be an integer"
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=out_size,  # type: ignore
                interpolation=getattr(self.args, "interpolation", transforms_F.InterpolationMode.BICUBIC),
                antialias=True,
            )
            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict


class ResizeLargestSide(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs resizing to larger side

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            out_size = obtain_augmentation_size(data_dict, self.args)
            assert isinstance(out_size, int), "Arg size in resize should be an integer"
            orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)

            scaling_ratio = min(out_size / orig_w, out_size / orig_h)
            target_size = [int(scaling_ratio * orig_h), int(scaling_ratio * orig_w)]

            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=target_size,
                interpolation=getattr(self.args, "interpolation", transforms_F.InterpolationMode.BICUBIC),
                antialias=True,
            )
            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict


class ResizeSmallestSideAspectPreserving(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs aspect-ratio preserving resizing.
        Image is resized to the dimension which has the smaller ratio of (size / target_size).
        First we compute (w_img / w_target) and (h_img / h_target) and resize the image
        to the dimension that has the smaller of these ratios.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        img_size = obtain_augmentation_size(data_dict, self.args)
        assert isinstance(img_size, (tuple, omegaconf.listconfig.ListConfig)), (
            f"Arg size in resize should be a tuple, get {type(img_size)}, {img_size}"
        )
        img_w, img_h = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        scaling_ratio = max((img_w / orig_w), (img_h / orig_h))
        target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

        assert target_size[0] >= img_h and target_size[1] >= img_w, (
            f"Resize error. orig {(orig_w, orig_h)} desire {img_size} compute {target_size}"
        )

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=target_size,  # type: ignore
                interpolation=(
                    self.args["interpolation"]
                    if "interpolation" in self.args
                    else transforms_F.InterpolationMode.BICUBIC
                ),
                antialias=True,
            )

            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict


class ResizeLargestSideAspectPreserving(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs aspect-ratio preserving resizing.
        Image is resized to the dimension which has the larger ratio of (size / target_size).
        First we compute (w_img / w_target) and (h_img / h_target) and resize the image
        to the dimension that has the larger of these ratios.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        img_size = obtain_augmentation_size(data_dict, self.args)
        assert isinstance(img_size, (tuple, omegaconf.listconfig.ListConfig)), (
            f"Arg size in resize should be a tuple, get {type(img_size)}, {img_size}"
        )
        img_w, img_h = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        scaling_ratio = min((img_w / orig_w), (img_h / orig_h))
        target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

        assert target_size[0] <= img_h and target_size[1] <= img_w, (
            f"Resize error. orig {(orig_w, orig_h)} desire {img_size} compute {target_size}"
        )

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=target_size,  # type: ignore
                interpolation=getattr(self.args, "interpolation", transforms_F.InterpolationMode.BICUBIC),
                antialias=True,
            )

            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict
