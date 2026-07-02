# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import atexit
import os
import sys
from typing import Any

import torch.distributed as dist
from loguru._logger import Core, Logger

RANK0_ONLY = True
LEVEL = os.environ.get("LOGURU_LEVEL", "INFO")
RANK = int(os.environ.get("RANK", "0"))


def make_new_logger(depth: int = 1) -> Logger:
    return Logger(
        core=Core(),
        exception=None,
        depth=depth,
        record=False,
        lazy=False,
        colors=False,
        raw=False,
        capture=True,
        patchers=[],
        extra={},
    )


logger = make_new_logger(depth=1)
atexit.register(logger.remove)


def _add_relative_path(record: dict[str, Any]) -> None:
    try:
        start = os.getcwd()
        record["extra"]["relative_path"] = os.path.relpath(record["file"].path, start)
    except OSError:
        # CWD may have been removed (e.g. on some ranks in distributed jobs).
        # Fall back to the absolute path so logging still works.
        record["extra"]["relative_path"] = f"<cwd-unavailable>:{record['file'].path}"


*options, _, extra = logger._options  # type: ignore
logger._options = tuple([*options, [_add_relative_path], extra])  # type: ignore


def init_loguru_stdout() -> None:
    logger.remove()
    datetime_format = get_datetime_format()
    machine_format = get_machine_format()
    message_format = get_message_format()
    logger.add(
        sys.stdout,
        level=LEVEL,
        format=f"{datetime_format}{machine_format}{message_format}",
        filter=_rank0_only_filter,
    )


def init_loguru_file(path: str) -> None:
    datetime_format = get_datetime_format()
    machine_format = get_machine_format()
    message_format = get_message_format()
    logger.add(
        path,
        encoding="utf8",
        level=LEVEL,
        format=f"{datetime_format}{machine_format}{message_format}",
        rotation="100 MB",
        filter=lambda result: _rank0_only_filter(result) or not RANK0_ONLY,
        enqueue=True,
    )


def get_datetime_format() -> str:
    return "[<green>{time:MM-DD HH:mm:ss}</green>|"


def get_machine_format() -> str:
    node_id = os.environ.get("NGC_ARRAY_INDEX", "0")
    num_nodes = int(os.environ.get("NGC_ARRAY_SIZE", "1"))
    machine_format = ""
    rank = 0
    if dist.is_available():
        if not RANK0_ONLY and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            machine_format = (
                f"<red>[Node{node_id:<3}/{num_nodes:<3}][RANK{rank:<5}/{world_size:<5}]" + "[{process.name:<8}]</red>| "
            )
    return machine_format


def get_message_format() -> str:
    message_format = "<level>{level}</level>|<cyan>{extra[relative_path]}:{line}:{function}</cyan>] {message}"
    return message_format


def _rank0_only_filter(record: Any) -> bool:
    is_rank0 = record["extra"].get("rank0_only", True)
    if RANK == 0 and is_rank0:
        return True
    if not is_rank0:
        record["message"] = f"[RANK {RANK}] " + record["message"]
    return not is_rank0


def _prepare_log_message(
    message: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    if args:
        try:
            return message % args, (), kwargs
        except (TypeError, ValueError):
            pass
    if kwargs:
        try:
            return message % kwargs, (), {}
        except (KeyError, TypeError, ValueError):
            pass
    return message, args, kwargs


def _log_with_rank(
    level: str,
    message: str,
    *args: Any,
    rank0_only: bool = True,
    exc_info: Any = None,
    **kwargs: Any,
) -> None:
    message, args, kwargs = _prepare_log_message(message, args, kwargs)
    bound_logger = logger.opt(depth=1, exception=exc_info).bind(rank0_only=rank0_only)
    getattr(bound_logger, level)(message, *args, **kwargs)


def trace(message: str, *args: Any, rank0_only: bool = True, exc_info: Any = None, **kwargs: Any) -> None:
    _log_with_rank("trace", message, *args, rank0_only=rank0_only, exc_info=exc_info, **kwargs)


def debug(message: str, *args: Any, rank0_only: bool = True, exc_info: Any = None, **kwargs: Any) -> None:
    _log_with_rank("debug", message, *args, rank0_only=rank0_only, exc_info=exc_info, **kwargs)


def info(message: str, *args: Any, rank0_only: bool = True, exc_info: Any = None, **kwargs: Any) -> None:
    _log_with_rank("info", message, *args, rank0_only=rank0_only, exc_info=exc_info, **kwargs)


def success(message: str, *args: Any, rank0_only: bool = True, exc_info: Any = None, **kwargs: Any) -> None:
    _log_with_rank("success", message, *args, rank0_only=rank0_only, exc_info=exc_info, **kwargs)


def warning(message: str, *args: Any, rank0_only: bool = True, exc_info: Any = None, **kwargs: Any) -> None:
    _log_with_rank("warning", message, *args, rank0_only=rank0_only, exc_info=exc_info, **kwargs)


def error(message: str, *args: Any, rank0_only: bool = True, exc_info: Any = None, **kwargs: Any) -> None:
    _log_with_rank("error", message, *args, rank0_only=rank0_only, exc_info=exc_info, **kwargs)


def critical(message: str, *args: Any, rank0_only: bool = True, exc_info: Any = None, **kwargs: Any) -> None:
    _log_with_rank("critical", message, *args, rank0_only=rank0_only, exc_info=exc_info, **kwargs)


def exception(message: str, *args: Any, rank0_only: bool = True, exc_info: Any = True, **kwargs: Any) -> None:
    _log_with_rank("exception", message, *args, rank0_only=rank0_only, exc_info=exc_info, **kwargs)


# Execute at import time.
init_loguru_stdout()
