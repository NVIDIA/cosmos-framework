# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import csv
from io import StringIO

from cosmos_framework.utils.easy_io.handlers.base import BaseFileHandler


class CsvHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        del kwargs
        reader = csv.reader(file)
        return list(reader)

    def dump_to_fileobj(self, obj, file, **kwargs):
        del kwargs
        writer = csv.writer(file)
        if not all(isinstance(row, list) for row in obj):
            raise ValueError("Each row must be a list")
        writer.writerows(obj)

    def dump_to_str(self, obj, **kwargs):
        del kwargs
        output = StringIO()
        writer = csv.writer(output)
        if not all(isinstance(row, list) for row in obj):
            raise ValueError("Each row must be a list")
        writer.writerows(obj)
        return output.getvalue()
