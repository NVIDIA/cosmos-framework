# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pandas as pd

from cosmos_framework.utils.easy_io.handlers.base import BaseFileHandler  # isort:skip


class PandasHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        return pd.read_csv(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        obj.to_csv(file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError("PandasHandler does not support dumping to str")
