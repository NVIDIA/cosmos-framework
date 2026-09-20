# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Torch JPEG compression round trip that runs on any device.

Adapted from DiffJPEG (MIT) https://github.com/mlomnitz/DiffJPEG through BasicSR and the
Cosmos Transfer1 corruptors. Changes versus the Transfer1 copy: constant tables are buffers
instead of parameters, nothing is pinned to CUDA or bfloat16, and quality can be a per-frame
tensor. Padding to a multiple of 16 handles sizes that are not divisible by 8:
https://dsp.stackexchange.com/questions/35339/jpeg-dct-padding/35343#35343
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

_Y_TABLE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
).T  # [8,8]

_C_TABLE = np.full((8, 8), 99, dtype=np.float32)  # [8,8]
_C_TABLE[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]]).T


def quality_to_factor(quality: torch.Tensor) -> torch.Tensor:  # quality: [B] in (0,100], returns [B]
    """Map JPEG quality to the quantisation-table multiplier (libjpeg convention)."""
    low = 5000.0 / quality  # [B]
    high = 200.0 - quality * 2  # [B]
    return torch.where(quality < 50, low, high) / 100.0  # [B]


def _dct_matrix() -> np.ndarray:  # returns [8,8], C[u,x] = cos((2x+1) u pi / 16)
    u = np.arange(8, dtype=np.float32).reshape(8, 1)  # [8,1]
    x = np.arange(8, dtype=np.float32).reshape(1, 8)  # [1,8]
    return np.cos((2 * x + 1) * u * np.pi / 16).astype(np.float32)  # [8,8]


def _alpha_outer() -> np.ndarray:  # returns [8,8]
    alpha = np.array([1.0 / np.sqrt(2)] + [1] * 7)  # [8]
    return np.outer(alpha, alpha).astype(np.float32)  # [8,8]


def _block_split(image: torch.Tensor) -> torch.Tensor:  # image: [B,H,W], returns [B,H*W/64,8,8]
    batch_size, height, _ = image.shape
    blocks = image.view(batch_size, height // 8, 8, -1, 8)  # [B,H/8,8,W/8,8]
    blocks = blocks.permute(0, 1, 3, 2, 4)  # [B,H/8,W/8,8,8]
    return blocks.contiguous().view(batch_size, -1, 8, 8)  # [B,H*W/64,8,8]


def _block_merge(blocks: torch.Tensor, height: int, width: int) -> torch.Tensor:  # blocks: [B,N,8,8], returns [B,H,W]
    batch_size = blocks.shape[0]
    image = blocks.view(batch_size, height // 8, width // 8, 8, 8)  # [B,H/8,W/8,8,8]
    image = image.permute(0, 1, 3, 2, 4)  # [B,H/8,8,W/8,8]
    return image.contiguous().view(batch_size, height, width)  # [B,H,W]


class DiffJPEG(nn.Module):
    """JPEG encode-decode simulator with 4:2:0 chroma subsampling.

    Args:
        differentiable: use a smooth rounding surrogate instead of ``torch.round``. Degradation
            for training data does not need gradients, so the default is hard rounding.
    """

    def __init__(self, differentiable: bool = False) -> None:
        super().__init__()
        self.differentiable = differentiable
        rgb2ycc = np.array(
            [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]], dtype=np.float32
        ).T  # [3,3]
        ycc2rgb = np.array([[1.0, 0.0, 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]], dtype=np.float32).T  # [3,3]
        self.register_buffer("rgb2ycc", torch.from_numpy(rgb2ycc), persistent=False)  # [3,3]
        self.register_buffer("ycc2rgb", torch.from_numpy(ycc2rgb), persistent=False)  # [3,3]
        self.register_buffer("ycc_shift", torch.tensor([0.0, 128.0, 128.0]), persistent=False)  # [3]
        self.register_buffer("y_table", torch.from_numpy(_Y_TABLE), persistent=False)  # [8,8]
        self.register_buffer("c_table", torch.from_numpy(_C_TABLE), persistent=False)  # [8,8]
        self.register_buffer("dct_mat", torch.from_numpy(_dct_matrix()), persistent=False)  # [8,8]
        self.register_buffer("alpha", torch.from_numpy(_alpha_outer()), persistent=False)  # [8,8]

    def _round(self, x: torch.Tensor) -> torch.Tensor:  # x: [...], returns [...]
        if self.differentiable:
            return torch.round(x) + (x - torch.round(x)) ** 3
        return torch.round(x)

    def _quantize(self, blocks: torch.Tensor, table: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        # blocks: [B,N,8,8]; table: [8,8]; factor: [B]; returns [B,N,8,8]
        scaled_table = table[None, None] * factor.view(-1, 1, 1, 1)  # [B,1,8,8]
        return self._round(blocks / scaled_table)  # [B,N,8,8]

    def _dequantize(self, blocks: torch.Tensor, table: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        # blocks: [B,N,8,8]; table: [8,8]; factor: [B]; returns [B,N,8,8]
        scaled_table = table[None, None] * factor.view(-1, 1, 1, 1)  # [B,1,8,8]
        return blocks * scaled_table  # [B,N,8,8]

    def _forward_dct(self, plane: torch.Tensor) -> torch.Tensor:  # plane: [B,H,W], returns [B,H*W/64,8,8]
        # Separable form of the 4D basis contraction: Y = C X C^T with C[u,x] = cos((2x+1) u pi / 16).
        blocks = _block_split(plane) - 128  # [B,N,8,8]
        return 0.25 * self.alpha * (self.dct_mat @ blocks @ self.dct_mat.T)  # [B,N,8,8]

    def _inverse_dct(self, blocks: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # blocks: [B,N,8,8], returns [B,H,W]
        blocks = blocks * self.alpha  # [B,N,8,8]
        blocks = 0.25 * (self.dct_mat.T @ blocks @ self.dct_mat) + 128  # [B,N,8,8]
        return _block_merge(blocks, height, width)  # [B,H,W]

    def forward(self, x: torch.Tensor, quality: torch.Tensor | float) -> torch.Tensor:
        """Compress and decompress a batch of RGB frames.

        Args:
            x: frames in [0, 1], shape ``[B,3,H,W]``, any float dtype.
            quality: JPEG quality in (0, 100], scalar or ``[B]`` tensor.

        Returns:
            Reconstructed frames in [0, 1], shape ``[B,3,H,W]``, dtype of ``x``.
        """
        batch_size, _, height, width = x.shape
        in_dtype = x.dtype
        x = x.float()  # [B,3,H,W]
        quality_t = torch.as_tensor(quality, dtype=torch.float32, device=x.device).reshape(-1)  # [1] or [B]
        if quality_t.numel() == 1:
            quality_t = quality_t.expand(batch_size)  # [B]
        factor = quality_to_factor(quality_t)  # [B]

        h_pad = (16 - height % 16) % 16
        w_pad = (16 - width % 16) % 16
        x = F.pad(x, (0, w_pad, 0, h_pad), mode="constant", value=0)  # [B,3,Hp,Wp]
        padded_h, padded_w = height + h_pad, width + w_pad

        ycc = torch.tensordot(x.permute(0, 2, 3, 1) * 255.0, self.rgb2ycc, dims=1) + self.ycc_shift  # [B,Hp,Wp,3]
        y = ycc[..., 0]  # [B,Hp,Wp]
        cb = F.avg_pool2d(ycc[..., 1:2].permute(0, 3, 1, 2), kernel_size=2, stride=2)[:, 0]  # [B,Hp/2,Wp/2]
        cr = F.avg_pool2d(ycc[..., 2:3].permute(0, 3, 1, 2), kernel_size=2, stride=2)[:, 0]  # [B,Hp/2,Wp/2]

        y_q = self._quantize(self._forward_dct(y), self.y_table, factor)  # [B,N,8,8]
        cb_q = self._quantize(self._forward_dct(cb), self.c_table, factor)  # [B,N/4,8,8]
        cr_q = self._quantize(self._forward_dct(cr), self.c_table, factor)  # [B,N/4,8,8]

        y_rec = self._inverse_dct(self._dequantize(y_q, self.y_table, factor), padded_h, padded_w)  # [B,Hp,Wp]
        cb_rec = self._inverse_dct(
            self._dequantize(cb_q, self.c_table, factor), padded_h // 2, padded_w // 2
        )  # [B,Hp/2,Wp/2]
        cr_rec = self._inverse_dct(
            self._dequantize(cr_q, self.c_table, factor), padded_h // 2, padded_w // 2
        )  # [B,Hp/2,Wp/2]
        cb_up = cb_rec.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)  # [B,Hp,Wp]
        cr_up = cr_rec.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)  # [B,Hp,Wp]

        ycc_rec = torch.stack([y_rec, cb_up, cr_up], dim=-1) - self.ycc_shift  # [B,Hp,Wp,3]
        rgb = torch.tensordot(ycc_rec, self.ycc2rgb, dims=1).permute(0, 3, 1, 2)  # [B,3,Hp,Wp]
        rgb = rgb.clamp(0.0, 255.0) / 255.0  # [B,3,Hp,Wp]
        return rgb[:, :, :height, :width].to(in_dtype)  # [B,3,H,W]
