# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Functions for performing operations with broadcasting to the right axis
#
# Example
# input1: tensor of size (N1, N2)
# input2: tensor of size (N1, N2, N3, N4)
# batch_mul(input1, input2) = input1[:, :, None, None] * input2
#
# If the common dimensions don't match, we raise an assertion error.

from torch import Tensor


def common_broadcast(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    ndims1 = x.ndim
    ndims2 = y.ndim

    common_ndims = min(ndims1, ndims2)
    for axis in range(common_ndims):
        assert x.shape[axis] == y.shape[axis], "Dimensions not equal at axis {}".format(axis)

    if ndims1 < ndims2:
        x = x.reshape(x.shape + (1,) * (ndims2 - ndims1))  # x broadcast-padded to ndims2: [*x.shape,1,...]
    elif ndims2 < ndims1:
        y = y.reshape(y.shape + (1,) * (ndims1 - ndims2))  # y broadcast-padded to ndims1: [*y.shape,1,...]

    return x, y


def batch_add(x: Tensor, y: Tensor) -> Tensor:
    x, y = common_broadcast(x, y)
    return x + y  # broadcast result shape


def batch_mul(x: Tensor, y: Tensor) -> Tensor:
    x, y = common_broadcast(x, y)
    return x * y  # broadcast result shape


def batch_sub(x: Tensor, y: Tensor) -> Tensor:
    x, y = common_broadcast(x, y)
    return x - y  # broadcast result shape


def batch_div(x: Tensor, y: Tensor) -> Tensor:
    x, y = common_broadcast(x, y)
    return x / y  # broadcast result shape
