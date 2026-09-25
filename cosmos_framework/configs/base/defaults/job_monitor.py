# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Job monitoring shared by VLM and VFM without importing either model stack."""

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.callbacks.device_monitor import DeviceMonitor
from cosmos_framework.callbacks.heart_beat import HeartBeat
from cosmos_framework.callbacks.wall_clock_checkpoint import WallClockCheckpoint

JOB_MONITOR_CALLBACKS = dict(
    heart_beat=L(HeartBeat)(
        every_n=200,
        update_interval_in_minute=20,
        save_s3="${upload_reproducible_setup}",
    ),
    device_monitor=L(DeviceMonitor)(
        every_n=200,
        save_s3="${upload_reproducible_setup}",
        upload_every_n_mul=5,
    ),
    # Interval comes from the environment, so submitting to a cluster that enforces a
    # wall-clock bound turns this on without every experiment config opting in.
    wall_clock_checkpoint=L(WallClockCheckpoint)(),
)
