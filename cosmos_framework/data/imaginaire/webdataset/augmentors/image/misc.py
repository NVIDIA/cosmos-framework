# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Union

import torch
from PIL import Image


def obtain_image_size(data_dict: dict, input_keys: list) -> tuple[int, int]:
    r"""Function for obtaining the image size from the data dict.

    Args:
        data_dict (dict): Input data dict
        input_keys (list): List of input keys
    Returns:
        width (int): Width of the input image
        height (int): Height of the input image
    """

    data1 = data_dict[input_keys[0]]
    if isinstance(data1, Image.Image):
        width, height = data1.size
    elif isinstance(data1, torch.Tensor):
        height, width = data1.size()[-2:]
    else:
        raise ValueError("data to random crop should be PIL Image or tensor")

    return width, height


def obtain_augmentation_size(data_dict: dict, augmentor_cfg: dict) -> Union[int, tuple]:
    r"""Function for obtaining size of the augmentation.
    When dealing with multi-aspect ratio dataloaders, we need to
    find the augmentation size from the aspect ratio of the data.
    If data_dict contains "_res_size_map" (e.g. from resolution sampling),
    that map is used instead of augmentor_cfg["size"].

    Args:
        data_dict (dict): Input data dict
        augmentor_cfg (dict): Augmentor config
    Returns:
        aug_size (int): Size of augmentation
    """
    if "__url__" in data_dict and "aspect_ratio" in data_dict["__url__"].meta.opts:
        aspect_ratio = data_dict["__url__"].meta.opts["aspect_ratio"]
    else:  # Non-webdataset format
        aspect_ratio = data_dict["aspect_ratio"]
    if "_res_size_map" in data_dict:
        return data_dict["_res_size_map"][aspect_ratio]
    return augmentor_cfg["size"][aspect_ratio]
