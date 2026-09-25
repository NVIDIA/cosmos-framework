# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dtype coercion for tokenizer configs that have been through a config round trip."""

import torch

__all__ = ["as_torch_dtype"]


def as_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Return ``dtype`` as a ``torch.dtype``, accepting the serialized string form.

    The LiDAR tokenizer configs pin a real ``torch.dtype`` (``_LIDAR_INFERENCE_DTYPE``
    in configs/base/defaults/tokenizer.py). Serializing a model config to ``config.json``
    turns that into the string ``"float32"``, and nothing on the way back in turns it
    into a dtype again -- so an exported checkpoint reaches ``Module.to(dtype=...)``
    with a ``str`` and dies with::

        TypeError: to() received an invalid combination of arguments -
          got (dtype=str, device=torch.device, )

    ``device`` never had this problem because the constructors already normalize it
    through ``torch.device(...)``, which accepts a string. This is the dtype equivalent.

    Note ``torch.dtype("float32")`` does NOT work -- it raises
    ``TypeError: cannot create 'torch.dtype' instances`` -- so the name is resolved as
    an attribute instead. Both ``"float32"`` and ``"torch.float32"`` are accepted,
    because serializers differ on whether they keep the module prefix.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if not isinstance(dtype, str):
        raise TypeError(f"Expected a torch.dtype or its name, got {type(dtype).__name__}: {dtype!r}")

    resolved = getattr(torch, dtype.split(".")[-1], None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Not a torch dtype name: {dtype!r}")
    return resolved
