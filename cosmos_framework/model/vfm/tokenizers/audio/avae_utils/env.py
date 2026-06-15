# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.

from typing import Any


class AttrDict(dict):
    def __init__(self: "AttrDict", *args: Any, **kwargs: Any) -> None:
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
