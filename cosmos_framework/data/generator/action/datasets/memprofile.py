# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""No-op stub of imaginaire4's memory-profiling helpers.

The ported BaseActionLeRobotDataset imports these for optional RSS tracking;
in cosmos-framework we disable profiling entirely (memprofile_enabled -> False),
so every symbol here is a cheap no-op with the same call signature.
"""

from contextlib import contextmanager


def memprofile_enabled() -> bool:
    return False


def deep_size(obj, *args, **kwargs) -> int:
    return 0


def fmt_mb(n, *args, **kwargs) -> str:
    return "n/a"


def log_worker_memory_breakdown(*args, **kwargs) -> None:
    return None


@contextmanager
def rss_tracker(*args, **kwargs):
    yield
