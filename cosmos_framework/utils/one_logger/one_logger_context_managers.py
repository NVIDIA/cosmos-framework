# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from contextlib import contextmanager
from typing import Generator

from cosmos_framework.utils.log import logger
from cosmos_framework.utils.one_logger.one_logger_utils import get_one_logger, one_logger_is_initialized


@contextmanager
def data_loader_init() -> Generator[None, None, None]:
    """
    Wrap the execution data loader initialization by invoking the one logger callbacks.
    """
    try:
        one_logger = get_one_logger()
        if one_logger_is_initialized():
            one_logger.on_dataloader_init_start()

        yield

    finally:
        try:
            if one_logger_is_initialized():
                one_logger.on_dataloader_init_end()
        except Exception as exc:  # noqa: BLE001
            logger.warning("one_logger.on_dataloader_init_end() failed (non-fatal): %s", exc)


@contextmanager
def model_init(set_barrier: bool = False) -> Generator[None, None, None]:
    """
    Wrap the instantiation of the model by invoking the one logger callbacks.
    """
    try:
        one_logger = get_one_logger()
        if one_logger_is_initialized():
            one_logger.on_model_init_start(set_barrier=set_barrier)

        yield

    finally:
        try:
            if one_logger_is_initialized():
                one_logger.on_model_init_end(set_barrier=set_barrier)
        except Exception as exc:  # noqa: BLE001
            logger.warning("one_logger.on_model_init_end() failed (non-fatal): %s", exc)
