# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import gzip
import pickle
from io import BytesIO
from typing import Any

from cosmos_framework.utils.easy_io.handlers.pickle_handler import PickleHandler


class GzipHandler(PickleHandler):
    str_like = False

    def load_from_fileobj(self, file: BytesIO, **kwargs):
        with gzip.GzipFile(fileobj=file, mode="rb") as f:
            return pickle.load(f)

    def dump_to_fileobj(self, obj: Any, file: BytesIO, **kwargs):
        with gzip.GzipFile(fileobj=file, mode="wb") as f:
            pickle.dump(obj, f)
