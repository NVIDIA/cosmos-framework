# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Optional

import torch
import torchvision.transforms.functional as transforms_F
from loguru import logger as logging

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.data.imaginaire.webdataset.augmentors.image.misc import obtain_augmentation_size, obtain_image_size


class CenterCrop(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs center crop.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
            We also save the cropping parameters in the aug_params dict
            so that it will be used by other transforms.
        """
        assert (self.args is not None) and ("size" in self.args), "Please specify size in args"

        img_size = obtain_augmentation_size(data_dict, self.args)
        width, height = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        for key in self.input_keys:
            data_dict[key] = transforms_F.center_crop(data_dict[key], [height, width])

        # We also add the aug params we use. This will be useful for other transforms
        crop_x0 = (orig_w - width) // 2
        crop_y0 = (orig_h - height) // 2
        cropping_params = {
            "resize_w": orig_w,
            "resize_h": orig_h,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_w": width,
            "crop_h": height,
        }

        if "aug_params" not in data_dict:
            data_dict["aug_params"] = dict()

        data_dict["aug_params"]["cropping"] = cropping_params
        return data_dict


class BottomCrop(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Crops rows from the bottom of the image/video to reach ``target_height``.

        The top of the frame is preserved (content is top-anchored). Width is unchanged.
        Works for 3-D ``[C, H, W]`` images and 4-D ``[C, T, H, W]`` or ``[T, C, H, W]``
        videos — the last two dims are always treated as (H, W).

        Args:
            data_dict (dict): Input data dict. ``self.args["target_height"]`` is the
                desired output height. Source height must be ``>= target_height``.

        Returns:
            data_dict (dict): Output dict where images are bottom-cropped and
            ``image_size`` is updated to ``[target_h, w, orig_h, orig_w]`` to mirror
            :class:`ReflectionPadding`'s contract.
        """
        assert (self.args is not None) and ("target_height" in self.args), "Please specify target_height in args"
        if self.output_keys is None:
            self.output_keys = self.input_keys

        target_h = int(self.args["target_height"])
        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        assert orig_h >= target_h, (
            f"BottomCrop requires source height >= target_height: got orig_h={orig_h}, target_h={target_h}"
        )

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            tensor = data_dict[inp_key]
            # Slice the last 2 dims; the second-to-last dim is height regardless of
            # whether the tensor is CHW, CTHW, or TCHW.
            data_dict[out_key] = tensor[..., :target_h, :]

            if out_key != inp_key:
                del data_dict[inp_key]

        data_dict["image_size"] = torch.tensor([target_h, orig_w, orig_h, orig_w], dtype=torch.float)

        return data_dict


class RandomCrop(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs random crop.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
            We also save the cropping parameters in the aug_params dict
            so that it will be used by other transforms.
        """

        img_size = obtain_augmentation_size(data_dict, self.args)
        width, height = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        # Obtaining random crop coords
        try:
            crop_x0 = int(torch.randint(0, orig_w - width + 1, size=(1,)).item())
            crop_y0 = int(torch.randint(0, orig_h - height + 1, size=(1,)).item())
        except Exception as e:
            logging.warning(
                f"Random crop failed. Performing center crop, original_size(wxh): {orig_w}x{orig_h}, random_size(wxh): {width}x{height}"
            )
            for key in self.input_keys:
                data_dict[key] = transforms_F.center_crop(data_dict[key], [height, width])
            crop_x0 = (orig_w - width) // 2
            crop_y0 = (orig_h - height) // 2

        # We also add the aug params we use. This will be useful for other transforms
        cropping_params = {
            "resize_w": orig_w,
            "resize_h": orig_h,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_w": width,
            "crop_h": height,
        }

        if "aug_params" not in data_dict:
            data_dict["aug_params"] = dict()

        data_dict["aug_params"]["cropping"] = cropping_params

        # We must perform same random cropping for all input keys
        for key in self.input_keys:
            data_dict[key] = transforms_F.crop(data_dict[key], crop_y0, crop_x0, height, width)
        return data_dict
