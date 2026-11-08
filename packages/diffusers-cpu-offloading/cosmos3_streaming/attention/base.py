# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class CosmosAttentionBackend(Protocol):
    """Attention contract used by the split Cosmos executor.

    Inputs and outputs use token-major NHD layout: ``[tokens, heads, dim]``.
    The backend must support GQA, where K/V have fewer heads than Q.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        """Return attention output in the same NHD layout as ``q``."""
