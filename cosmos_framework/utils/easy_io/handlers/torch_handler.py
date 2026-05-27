# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

try:
    import torch
except ImportError:
    torch = None

from cosmos_framework.utils.easy_io.handlers.base import BaseFileHandler


class TorchHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        return torch.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        torch.save(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError
