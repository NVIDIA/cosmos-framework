# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""workers — WorkerGroup substrate: the WG / Replica / rank abstraction
(`worker_group`) plus its Ray realizations (`ray`, `replica`). Concrete role bodies
live in `roles`. (ray/replica are imported on demand, so importing this package
does not require Ray.)
"""

from cosmos_framework.posttrain.workers.worker_group import Handle, WorkerGroup

__all__ = ["WorkerGroup", "Handle"]
