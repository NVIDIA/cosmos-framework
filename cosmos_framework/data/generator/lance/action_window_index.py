# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionWindowLocation:
    """A runtime window's source-row identity and local anchor frame."""

    row_position: int  # index into the row-index cache
    row_id: int  # row_id tracked in the cache to refer back to source lance table
    local_start_frame: int  # start_frame local to a subtask clip
    video_frame_count: int | None = None  # truncated one-anchor windows carry their effective T+1 length
