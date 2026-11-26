# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native adapter construction; optional quantizers are imported only when selected."""

from __future__ import annotations

import warnings

# Native NF4 is tested with the bitsandbytes 0.49 series (the posttrain ``qlora`` dependency group).
BITSANDBYTES_MIN_VERSION = "0.49.2"
BITSANDBYTES_TESTED_SERIES = (0, 49)


def check_bitsandbytes_version(installed: str | None = None) -> str:
    """Require the minimum supported bitsandbytes and warn when it is newer than the tested series."""
    from packaging.version import Version

    if installed is None:
        from importlib.metadata import version

        installed = version("bitsandbytes")
    parsed = Version(installed)
    if parsed < Version(BITSANDBYTES_MIN_VERSION):
        raise ValueError(
            f"native NF4 requires bitsandbytes>={BITSANDBYTES_MIN_VERSION} (the posttrain qlora group installs "
            f"bitsandbytes=={BITSANDBYTES_MIN_VERSION}); found {installed}"
        )
    if parsed.release[:2] != BITSANDBYTES_TESTED_SERIES:
        warnings.warn(
            f"native NF4 is tested with bitsandbytes {'.'.join(map(str, BITSANDBYTES_TESTED_SERIES))}.x; "
            f"found {installed}",
            UserWarning,
            stacklevel=2,
        )
    return installed
