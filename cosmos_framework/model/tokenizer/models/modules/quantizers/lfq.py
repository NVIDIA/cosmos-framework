# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Lookup-Free Quantization (LFQ).

Proposed in https://arxiv.org/abs/2310.05737

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.

References:
    - https://github.com/lucidrains/vector-quantize-pytorch
    - https://github.com/theAdamColton/ijepa-enhanced
"""

from __future__ import annotations

from collections import namedtuple
from math import ceil, log2
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch.nn import Module

if TYPE_CHECKING:
    from cosmos_framework.model.tokenizer.models.modules.sparse_tensor import SparseTensor

__all__ = [
    "LFQ",
    "LossBreakdown",
]


LossBreakdown = namedtuple(
    "LossBreakdown",
    ["per_sample_entropy", "codebook_entropy", "commitment", "avg_probs"],
)

_MAX_DIM_ONLY_CODEBOOK_BITS = 20


# Helper functions


def exists(v):
    """Check if value exists (is not None)."""
    return v is not None


def default(*args):
    """Return first non-None value from args (calling if callable)."""
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def entropy(prob: torch.Tensor) -> torch.Tensor:
    """Compute entropy of probability distribution."""
    return (-prob * torch.log(prob + 1e-5)).sum(dim=-1)


def mult_along_first_dims(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multiply x by y along leading dimensions of y."""
    ndim_to_expand = x.ndim - y.ndim
    for _ in range(ndim_to_expand):
        y = y.unsqueeze(-1)
    return x * y


def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Take mean of x elements not masked by m.

    The mean is taken along the shared leading dims of m.
    This is faster for torch-compile on batches than tensor indexing.
    """
    x = mult_along_first_dims(x, m)
    x = x / m.sum()
    return x.sum(tuple(range(m.ndim)))


def entropy_loss(
    logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 0.01,
    sample_minimization_weight: float = 1.0,
    batch_maximization_weight: float = 1.0,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute entropy loss for LFQ.

    The loss encourages:
        - LOW per-sample entropy (focused code usage per sample)
        - HIGH batch-level entropy (diverse code usage across batch)

    Formula: loss = sample_entropy - batch_entropy

    NEGATIVE loss is GOOD: means batch entropy > sample entropy,
    indicating good codebook utilization with focused per-sample usage.

    Reference:
        https://github.com/google-research/magvit/
        LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)

    Args:
        logits: Affinities over the last dimension.
        mask: Optional mask for selective processing.
        temperature: Softmax temperature.
        sample_minimization_weight: Weight for per-sample entropy term.
        batch_maximization_weight: Weight for batch entropy term.
        eps: Small epsilon for numerical stability.

    Returns:
        Tuple of (per_sample_entropy, batch_entropy, loss).
    """
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + eps, -1)

    if mask is not None:
        avg_probs = masked_mean(probs, mask)
    else:
        avg_probs = reduce(probs, "... D -> D", "mean")

    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    if mask is not None:
        sample_entropy = masked_mean(sample_entropy, mask).mean()
    else:
        sample_entropy = torch.mean(sample_entropy)

    # Key insight: loss = sample_entropy - batch_entropy
    # Negative loss is DESIRABLE (batch entropy > sample entropy)
    loss = (sample_minimization_weight * sample_entropy) - (batch_maximization_weight * avg_entropy)

    return sample_entropy, avg_entropy, loss


class LFQ(Module):
    """Lookup-Free Quantization module.

    Quantizes continuous values to binary codes {-1, 1} per dimension,
    with entropy loss to encourage codebook utilization.
    """

    def __init__(
        self,
        *,
        dim: int | None = None,
        codebook_size: int | None = None,
        num_codebooks: int = 1,
        sample_minimization_weight: float = 1.0,
        batch_maximization_weight: float = 1.0,
        token_factorization: bool = False,
        factorized_bits: list[int] = [9, 9],
    ) -> None:
        """Initialize LFQ.

        Args:
            dim: Input feature dimension (defaults to log2(codebook_size)).
            codebook_size: Size of codebook (must be power of 2).
            num_codebooks: Number of codebooks to use.
            sample_minimization_weight: Weight for per-sample entropy in loss.
            batch_maximization_weight: Weight for batch entropy in loss.
            token_factorization: Whether to use factorized tokens.
            factorized_bits: Bit split for factorized tokens.
        """
        super().__init__()

        if not exists(dim) and not exists(codebook_size):
            raise ValueError("Either dim or codebook_size must be specified for LFQ.")

        if codebook_size is None:
            if dim is None:
                raise ValueError("dim must be specified when codebook_size is omitted.")
            if dim > _MAX_DIM_ONLY_CODEBOOK_BITS:
                raise ValueError(
                    "LFQ dim-only construction materializes a 2**dim codebook; "
                    f"got dim={dim}. Pass codebook_size explicitly or use dim <= {_MAX_DIM_ONLY_CODEBOOK_BITS}."
                )
            resolved_codebook_size = 2**dim
        else:
            resolved_codebook_size = int(codebook_size)
        if not log2(resolved_codebook_size).is_integer():
            raise ValueError(f"codebook size must be power of 2 (suggested {2 ** ceil(log2(resolved_codebook_size))})")

        self.codebook_size = resolved_codebook_size
        self.codebook_dim = int(log2(self.codebook_size))

        codebook_dims = self.codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = dim != codebook_dims
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = self.codebook_dim
        self.num_codebooks = num_codebooks
        self.project_in = nn.Linear(self.dim, codebook_dims) if has_projections else nn.Identity()
        self.project_out = nn.Linear(codebook_dims, self.dim) if has_projections else nn.Identity()

        # For entropy loss
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

        # For token factorization
        self.token_factorization = token_factorization
        if not self.token_factorization:
            self.register_buffer("mask", 2 ** torch.arange(self.codebook_dim), persistent=False)
        else:
            self.factorized_bits = factorized_bits
            self.register_buffer("pre_mask", 2 ** torch.arange(factorized_bits[0]), persistent=False)
            self.register_buffer("post_mask", 2 ** torch.arange(factorized_bits[1]), persistent=False)

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # Build codebook
        all_codes = torch.arange(self.codebook_size)  # [K]
        bits = self.indices_to_bits(all_codes)  # [K,Z]
        codebook = bits * 2.0 - 1.0  # [K,Z]

        self.register_buffer("codebook", codebook, persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        """Return dtype of codebook."""
        return self.codebook.dtype

    def indices_to_bits(self, x: torch.Tensor) -> torch.Tensor:
        """Convert indices to big-endian bits.

        Args:
            x: Long tensor of indices.

        Returns:
            Boolean tensor of bits.
        """
        mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
        x = (x.unsqueeze(-1) & mask) != 0
        return x

    def get_codebook_entry(
        self,
        x: torch.Tensor,
        bhwc: tuple[int, int, int, int],
        order: str,
    ) -> torch.Tensor:
        """Get codebook entry for given indices.

        Args:
            x: Index tensor.
            bhwc: Tuple of (batch, height, width, channels).
            order: 'pre' or 'post' for factorized tokens.

        Returns:
            Decoded tensor.
        """
        if self.token_factorization:
            if order == "pre":
                mask = 2 ** torch.arange(self.factorized_bits[0], device=x.device, dtype=torch.long)
            else:
                mask = 2 ** torch.arange(self.factorized_bits[1], device=x.device, dtype=torch.long)
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)

        x = (x.unsqueeze(-1) & mask) != 0
        x = x * 2.0 - 1.0  # back to float
        b, h, w, c = bhwc
        x = rearrange(x, "b (h w) c -> b h w c", h=h, w=w, c=c)
        x = rearrange(x, "b h w c -> b c h w")
        return x

    def bits_to_indices(self, bits: torch.Tensor) -> torch.Tensor:
        """Convert big-endian bits to indices.

        Args:
            bits: Boolean tensor with bit dimension last.

        Returns:
            Long integer indices from 0 to codebook_size.
        """
        assert bits.shape[-1] == self.codebook_dim
        indices = 2 ** torch.arange(
            0,
            self.codebook_dim,
            1,
            dtype=torch.long,
            device=bits.device,
        )
        return (bits * indices).sum(-1)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode indices to continuous values.

        Args:
            x: Long tensor of codebook indices (..., NH) where NH is num_codebooks.

        Returns:
            Decoded tensor with values in {-1, 1}.
        """
        x = self.indices_to_bits(x)  # [...,NC,Z]
        x = x.to(self.dtype)  # [...,NC,Z]
        x = x * 2 - 1  # [...,NC,Z]
        x = rearrange(x, "... NC Z-> ... (NC Z)")  # [...,Dq]
        return self.project_out.to(x.dtype)(x)  # [...,D]

    def forward(
        self,
        x: "SparseTensor",
        inv_temperature: float = 100.0,
        return_loss_breakdown: bool = False,
        mask: torch.Tensor | None = None,
        return_loss: bool = True,
        fp32_loss_computation: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor]
        | tuple[tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor], LossBreakdown]
    ):
        """Forward pass for LFQ on SparseTensor.

        Args:
            x: SparseTensor input with .feats, .coords, .shape, .layout.
            inv_temperature: Inverse temperature for entropy loss softmax.
            return_loss_breakdown: Whether to return detailed loss components.
            mask: Optional mask for selective processing.
            return_loss: Whether to compute training losses.
            fp32_loss_computation: Whether to compute losses in fp32.

        Returns:
            Tuple of (quantized_feats, indices, quantizer_loss).
            If return_loss_breakdown, also returns LossBreakdown.
        """
        # Extract features from sparse tensor
        N, feature_dim = x.shape

        expected_dim = self.num_codebooks * self.codebook_dim
        if feature_dim != self.dim:
            raise ValueError(f"Feature dimension {feature_dim} doesn't match LFQ input dimension {self.dim}.")

        features = self.project_in.to(x.dtype)(x.view(N, feature_dim))  # [N,Dq]
        features_reshaped = features.view(N, self.num_codebooks, self.codebook_dim)  # [N,NC,Z]

        # Quantization step
        codebook_value = torch.tensor(1.0, device=x.device, dtype=x.dtype)  # []
        quantized_values = torch.where(features_reshaped > 0, codebook_value, -codebook_value)  # [N,NC,Z]

        # Index calculation
        if self.token_factorization:
            pre_bits = quantized_values[..., : self.factorized_bits[0]]  # [N,NC,Zpre]
            post_bits = quantized_values[..., self.factorized_bits[0] :]  # [N,NC,Zpost]

            indices_pre = ((pre_bits > 0).int() * self.pre_mask.int()).sum(-1)  # [N,NC]
            indices_post = ((post_bits > 0).int() * self.post_mask.int()).sum(-1)  # [N,NC]

            indices_pre_flat = indices_pre.flatten()  # [N*NC]
            indices_post_flat = indices_post.flatten()  # [N*NC]
            sparse_indices_quantized = (indices_pre_flat, indices_post_flat)
        else:
            indices = ((quantized_values > 0).int() * self.mask.int()).sum(-1)  # [N,NC]
            sparse_indices_quantized = indices.flatten()  # [N*NC]

        # Entropy loss (training only)
        if self.training and return_loss:
            if fp32_loss_computation:
                features_flat_fp32 = features_reshaped.view(-1, self.codebook_dim).float()  # [N*NC,Z]
                codebook_fp32 = self.codebook.float()  # [K,Z]
            else:
                features_flat_fp32 = features_reshaped.view(-1, self.codebook_dim)  # [N*NC,Z]
                codebook_fp32 = self.codebook.to(features_flat_fp32.dtype)  # [K,Z]

            logits = 2 * torch.mm(features_flat_fp32, codebook_fp32.T)  # [N*NC,K]

            if mask is not None:
                if mask.shape[0] != N:
                    raise ValueError(f"Mask shape {mask.shape} doesn't match number of features {N}")
                mask_expanded = mask.unsqueeze(1).repeat(1, self.num_codebooks).view(-1)  # [N*NC]
            else:
                mask_expanded = None

            temperature = 1.0 / inv_temperature if inv_temperature > 0 else 0.01
            per_sample_entropy, codebook_entropy, entropy_aux_loss = entropy_loss(
                logits=logits,
                mask=mask_expanded,
                temperature=temperature,
                sample_minimization_weight=self.sample_minimization_weight,
                batch_maximization_weight=self.batch_maximization_weight,
            )
        else:
            dtype = torch.float32 if fp32_loss_computation else x.dtype
            per_sample_entropy = torch.tensor(0.0, dtype=dtype, device=x.device)  # []
            codebook_entropy = torch.tensor(0.0, dtype=dtype, device=x.device)  # []
            entropy_aux_loss = torch.tensor(0.0, dtype=dtype, device=x.device)  # []

        # Commitment loss
        if self.training:
            if fp32_loss_computation:
                features_fp32 = features_reshaped.float()  # [N,NC,Z]
                quantized_fp32 = quantized_values.float()  # [N,NC,Z]
            else:
                features_fp32 = features_reshaped  # [N,NC,Z]
                quantized_fp32 = quantized_values  # [N,NC,Z]

            commit_loss = F.mse_loss(features_fp32, quantized_fp32.detach(), reduction="none")  # [N,NC,Z]

            if mask is not None:
                mask_expanded = mask.view(N, 1, 1).expand_as(commit_loss)  # [N,NC,Z]
                commit_loss = commit_loss[mask_expanded].mean()  # []
            else:
                commit_loss = commit_loss.mean()  # []
        else:
            dtype = torch.float32 if fp32_loss_computation else x.dtype
            commit_loss = torch.tensor(0.0, dtype=dtype, device=x.device)  # []

        # Straight-through estimator
        quantized_values_ste = features_reshaped + (quantized_values - features_reshaped).detach()  # [N,NC,Z]

        # Output construction
        quantized_feats = quantized_values_ste.view(N, expected_dim)  # [N,Dq]
        quantized_feats = self.project_out.to(x.dtype)(quantized_feats)  # [N,D]

        # Ensure fp32 losses if requested
        if self.training and return_loss and fp32_loss_computation:
            entropy_aux_loss = entropy_aux_loss.float()  # []
            per_sample_entropy = per_sample_entropy.float()  # []
            codebook_entropy = codebook_entropy.float()  # []
            commit_loss = commit_loss.float()  # []

        quantizer_loss = commit_loss + entropy_aux_loss  # []
        ret = (quantized_feats, sparse_indices_quantized, quantizer_loss)

        if not return_loss_breakdown:
            return ret

        placeholder_dtype = torch.float32 if fp32_loss_computation else x.dtype
        return ret, LossBreakdown(
            per_sample_entropy,
            codebook_entropy,
            commit_loss,
            torch.tensor(0.0, dtype=placeholder_dtype, device=x.device),  # []
        )
