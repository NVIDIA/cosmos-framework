# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Any, Union

from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.checkpoint.stateful import Stateful


class ModelWrapper(Stateful):
    """Wrapper for model state dict handling"""

    def __init__(self, model_parts: Union[list[nn.Module], nn.Module]):
        if not isinstance(model_parts, list):
            model_parts = [model_parts]
        self.model_parts = model_parts

    def state_dict(self) -> dict[str, Any]:
        sd = {}
        for model in self.model_parts:
            sd.update(get_model_state_dict(model))
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for model in self.model_parts:
            set_model_state_dict(
                model,
                model_state_dict=state_dict,
                options=StateDictOptions(strict=False),
            )
