# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

"""VRAD's active RopeRigPE, pinned to 979c4d32fe1da235c0a031ebf86d1767013db700.

Ported from genradar/networks/position_embedding.py and datasets/utils.py.
By default direction and moment are independently normalized as in that source.
The spatial ablation instead scales the raw moment by a fixed distance in metres.
The fixed frequencies introduce buffers only, with no learned parameters.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RopeRigPE(nn.Module):
    """Rotary Position Embedding from geometric (Plucker) features.

    Uses NeRF-style frequencies ``2^(linspace(min_freq_exp, max_freq_exp, n)) * pi`` so
    that input features in [-1, 1] produce angles in
    [2^min_freq_exp * pi, 2^max_freq_exp * pi]. The default [pi, 8 pi] matches the
    angular range of grid-index rope3d (~26).

    Each coordinate dimension gets ``total_angles // feature_dim`` unique
    frequencies. All dimensions are treated uniformly.
    """

    head_dim: int
    feature_dim: int
    min_freq_exponent: float
    max_freq_exponent: float
    freq_matrix: torch.Tensor  # [8,D/2]

    def __init__(
        self,
        head_dim: int,
        feature_dim: int = 8,
        max_freq_exponent: float = 3.0,
        min_freq_exponent: float = 0.0,
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 or feature_dim != 8:
            raise ValueError("RigRoPE requires a positive even head dimension and eight features")
        if not (math.isfinite(min_freq_exponent) and math.isfinite(max_freq_exponent)):
            raise ValueError("RigRoPE frequency exponents must be finite")
        if min_freq_exponent > max_freq_exponent:
            raise ValueError(
                f"RigRoPE min_freq_exponent={min_freq_exponent} exceeds max_freq_exponent={max_freq_exponent}"
            )
        self.head_dim = head_dim
        self.feature_dim = feature_dim
        self.min_freq_exponent = min_freq_exponent
        self.max_freq_exponent = max_freq_exponent
        freq_matrix = self._build_freq_matrix(
            feature_dim, head_dim // 2, max_freq_exponent, min_freq_exponent
        )  # [8,D/2]
        # This matrix is fully determined by the constructor arguments. Keeping it out of the
        # checkpoint lets a strict warm start from the pre-RigRoPE model initialize it locally.
        self.register_buffer("freq_matrix", freq_matrix, persistent=False)

    @staticmethod
    def _build_freq_matrix(
        feature_dim: int, total_angles: int, max_freq_exponent: float, min_freq_exponent: float = 0.0
    ) -> torch.Tensor:
        """Build NeRF-style ``[feature_dim, total_angles]`` frequency matrix.

        Angles are distributed evenly across all coordinate dimensions.
        Each dimension gets its own set of unique ``2^e * pi`` frequencies.
        """
        base_per_dim, extra = divmod(total_angles, feature_dim)
        freq_matrix = torch.zeros(feature_dim, total_angles)  # [8,D/2]
        col = 0
        for dimension in range(feature_dim):
            count = base_per_dim + int(dimension < extra)
            exponents = torch.linspace(min_freq_exponent, max_freq_exponent, count)  # [K]
            freq_matrix[dimension, col : col + count] = (2.0**exponents) * math.pi  # [K]
            col += count
        return freq_matrix.float()  # [8,D/2]

    def reset_parameters(self) -> None:
        """Re-initialise frequency buffer (needed after meta -> CUDA)."""
        new = self._build_freq_matrix(
            self.feature_dim, self.head_dim // 2, self.max_freq_exponent, self.min_freq_exponent
        )  # [8,D/2]
        self.freq_matrix.data.copy_(new.to(self.freq_matrix.device))  # [8,D/2]

    def forward(self, coords: torch.Tensor) -> torch.Tensor:  # [...,8] -> [N,1,1,D]
        """Compute TE-compatible RoPE; leading dimensions are flattened."""
        flat = coords.reshape(-1, self.feature_dim).float()  # [N,8]
        freq = self.freq_matrix.to(flat.device).float()  # [8,D/2]
        thetas = flat @ freq  # [N,D/2]
        emb = torch.cat([thetas, thetas], dim=-1)  # [N,D]
        return emb.view(flat.shape[0], 1, 1, self.head_dim).float()  # [N,1,1,D]


def rig_features(
    directions: torch.Tensor,
    origins: torch.Tensor,
    times: torch.Tensor,
    *,
    moment_scale_m: float | None = None,
    include_time: bool = True,
) -> torch.Tensor:  # directions, origins: [H,W,3]; times: [T]; returns [T,H,W,8]
    """Plucker features with zero depth and an optional continuous, metric moment.

    Disabling time keeps its existing frequency slots at zero; it does not
    redistribute those slots or alter the model's baseline mRoPE time encoding.
    """
    if moment_scale_m is not None and (not math.isfinite(moment_scale_m) or moment_scale_m <= 0):
        raise ValueError("RigRoPE moment scale must be finite and positive, in metres")
    d = directions / (torch.norm(directions, dim=-1, keepdim=True) + 1e-8)  # [H,W,3]
    m = torch.cross(origins.expand_as(d), d, dim=-1)  # [H,W,3]
    if moment_scale_m is None:
        m = m / (torch.norm(m, dim=-1, keepdim=True) + 1e-8)  # [H,W,3]
    else:
        m = m / moment_scale_m  # [H,W,3]
    depth = torch.zeros_like(d[..., :1])  # [H,W,1]
    spatial = torch.cat([d, m, depth], dim=-1)  # [H,W,7]
    spatial = spatial.unsqueeze(0).expand(len(times), -1, -1, -1)  # [T,H,W,7]
    encoded_times = times if include_time else torch.zeros_like(times)  # [T]
    timestamps = encoded_times[:, None, None, None].expand(-1, d.shape[0], d.shape[1], -1)  # [T,H,W,1]
    return torch.cat([spatial, timestamps], dim=-1)  # [T,H,W,8]


def resize_rig_features(coords: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """VRAD trilinear feature interpolation; deliberately no normalization after it."""
    # coords: [V,T,H,W,8]; each physical view has an independent interpolation grid.
    if tuple(coords.shape[1:4]) == shape:
        return coords  # [V,T,H,W,8]
    channels = coords.permute(0, 4, 1, 2, 3).float()  # [V,8,T,H,W]
    resized = F.interpolate(channels, size=shape, mode="trilinear", align_corners=False)  # [V,8,T',H',W']
    return resized.permute(0, 2, 3, 4, 1)  # [V,T',H',W',8]


def spread_over_even_pairs(angles: torch.Tensor) -> torch.Tensor:  # [N,D/2] -> [N,D]
    """Place a half-width ``RopeRigPE`` output on the even split-half pairs of a full head.

    ``RopeRigPE(D/2)`` pairs channel ``j`` with ``j + D/4``; its pair ``j`` lands on the full
    head's pair ``2j``, channels ``(2j, 2j + D/2)``. The odd pairs take angle zero, which leaves
    them unrotated for the caller to fill.
    """
    count, width = angles.shape
    if width % 4:
        raise ValueError(f"Even-pair RigRoPE needs a half-width angle table divisible by 4, got {width}")
    half = angles.new_zeros(count, width)  # [N,D/2], one angle per full-head pair
    half[:, 0::2] = angles[:, : width // 2]
    return torch.cat([half, half], dim=-1)  # [N,D]


def apply_rig_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply split-half rotary to Q or K; never to V."""
    # x: [N,H,D]; cos, sin: [N,D]
    first, second = x.chunk(2, dim=-1)  # [N,H,D/2], [N,H,D/2]
    rotated = torch.cat([-second, first], dim=-1)  # [N,H,D]
    return (x.float() * cos[:, None, :] + rotated.float() * sin[:, None, :]).to(x.dtype)  # [N,H,D]


def apply_mrope_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply baseline split-half mRoPE in its original dtype after the CP gather."""
    # x: [N,H,D]; cos, sin: [N,D_rotary], allowing the baseline's partial rotary embedding.
    rotary_dim = cos.shape[-1]
    first, second = x[..., :rotary_dim].chunk(2, dim=-1)  # each [N,H,D_rotary/2]
    rotated = torch.cat([-second, first], dim=-1)  # [N,H,D_rotary]
    output = x[..., :rotary_dim] * cos[:, None, :] + rotated * sin[:, None, :]  # [N,H,D_rotary]
    if rotary_dim == x.shape[-1]:
        return output  # [N,H,D]
    return torch.cat([output, x[..., rotary_dim:]], dim=-1)  # [N,H,D]
