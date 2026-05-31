# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0

import torch.nn as nn
from torch.nn import functional as F

from .filter import LowPassFilter1d, kaiser_sinc_filter1d


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    # x: [B,C,T]
    def forward(self, x):  # x: [B,C,T]
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")  # [B,C,T+2*pad]
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C
        )  # [B,C,T*ratio+pad_left+pad_right]
        x = x[..., self.pad_left : -self.pad_right]  # [B,C,T*ratio]

        return x  # [B,C,T*ratio]


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=self.kernel_size
        )

    def forward(self, x):  # x: [B,C,T]
        xx = self.lowpass(x)  # [B,C,T//ratio]

        return xx  # [B,C,T//ratio]
