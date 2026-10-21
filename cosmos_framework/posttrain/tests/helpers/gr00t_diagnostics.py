# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Scoped, observational captures of real GR00T inference for baseline calibration."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from typing import Any

import numpy as np
import torch


def cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu_copy(item) for item in value]
    return value


class Gr00tForwardCapture:
    """Observe collator/backbone/encoder/decoder seams and restore all hooks on exit."""

    def __init__(self, model: Any, collator_owner: Any, collator_attribute: str, processor: Any) -> None:
        self.model = model
        self.collator_owner = collator_owner
        self.collator_attribute = collator_attribute
        self.processor = processor
        self.data: dict[str, Any] = {"latents_before": [], "velocities": []}
        self._stack = ExitStack()

    def __enter__(self) -> Gr00tForwardCapture:
        original_collator = getattr(self.collator_owner, self.collator_attribute)
        original_decode = self.processor.decode_action

        def collate(*args: Any, **kwargs: Any) -> Any:
            value = original_collator(*args, **kwargs)
            self.data["collated_inputs"] = cpu_copy(value)
            return value

        def decode(*args: Any, **kwargs: Any) -> Any:
            self.data["normalized_action"] = cpu_copy(args[0])
            value = original_decode(*args, **kwargs)
            self.data["decoded_actions"] = cpu_copy(value)
            return value

        def backbone(_module: Any, _inputs: Any, output: Any) -> None:
            self.data["backbone_features"] = cpu_copy(output.backbone_features)
            self.data["backbone_buffers"] = cpu_copy(dict(_module.named_buffers()))

        def encoder(_module: Any, inputs: Any) -> None:
            self.data["latents_before"].append(cpu_copy(inputs[0]))

        def decoder(_module: Any, _inputs: Any, output: Any) -> None:
            self.data["velocities"].append(cpu_copy(output[:, -self.model.action_head.action_horizon :]))

        try:
            for owner, attribute, replacement in (
                (self.collator_owner, self.collator_attribute, collate),
                (self.processor, "decode_action", decode),
            ):
                owned, original = attribute in vars(owner), getattr(owner, attribute)
                setattr(owner, attribute, replacement)
                if owned:
                    self._stack.callback(setattr, owner, attribute, original)
                else:
                    self._stack.callback(delattr, owner, attribute)
            for module, method, callback in (
                (self.model.backbone, "register_forward_hook", backbone),
                (self.model.action_head.action_encoder, "register_forward_pre_hook", encoder),
                (self.model.action_head.action_decoder, "register_forward_hook", decoder),
            ):
                self._stack.callback(getattr(module, method)(callback).remove)
        except BaseException:
            self._stack.close()
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stack.close()
