# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""entrypoints — stable, per-algorithm runnable entries (the production launch path).

Each module exposes one algorithm's assembly and loop, run with
`python -m cosmos_framework.posttrain.entrypoints.<algorithm>` (e.g. `...entrypoints.dpo`).
Reusable mechanics live in the corresponding control, checkpoint, role, and utility packages.
"""
