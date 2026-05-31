# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Transfer control input augmentors (edge, blur, depth, seg) for cosmos3 VFM; copied from transfer2 to avoid cosmos dependency."""

from cosmos_framework.data.vfm.augmentors.transfer_control_input.control_input import AddControlInputComb

__all__ = ["AddControlInputComb"]
