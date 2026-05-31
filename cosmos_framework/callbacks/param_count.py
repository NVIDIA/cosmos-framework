# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Union

import torch.nn as nn

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.count_params import count_params
from cosmos_framework.utils.distributed import rank0_only
from cosmos_framework.utils.easy_io import easy_io


class ParamCount(Callback):
    def __init__(
        self,
        save_s3: bool = False,
    ):
        self.save_s3 = save_s3
        self.name = self.__class__.__name__

    @rank0_only
    def on_train_start(self, model: Union[ImaginaireModel, list[nn.Module]], iteration: int = 0) -> None:
        if isinstance(model, list):
            num_param = sum([count_params(m) for m in model])
        else:
            num_param = count_params(model)

        log.info(f"Total number of parameters on current rank: {num_param}", rank0_only=False)
        info = {
            "num_parameters": num_param,
        }

        if self.save_s3:
            rank = distributed.get_rank()
            easy_io.dump(info, f"s3://rundir/{self.name}_{rank}.yaml")
