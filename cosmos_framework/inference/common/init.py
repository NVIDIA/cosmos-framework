# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import atexit
import fnmatch
import logging
import os
import socket
import sys
import time
import warnings
from pathlib import Path

import loguru
import torch
from torch.distributed.elastic.multiprocessing.errors import get_error_handler

"""Script initialization."""


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def enable_distributed() -> bool:
    return get_world_size() > 1


def is_rank0() -> bool:
    return get_rank() == 0


def _get_logger_format() -> str:
    from cosmos_framework.utils import log

    # Inject job=<name> segment between datetime and machine prefixes. Loguru's
    # `extra[job_name]` is configured to default to "" via `_init_script`, then
    # updated to the real value by `init_output_dir` once config is loaded.
    job_format = "<yellow>job={extra[job_name]}</yellow>|"
    return f"{log.get_datetime_format()}{job_format}{log.get_machine_format()}{log.get_message_format()}"


_LOGGER_INCLUDE = [
    "cosmos_framework.model.attention",
    "cosmos_framework.utils.checkpoint_db",
    "imaginaire.trainer",
    "cosmos_framework.utils.generator.model_loader",
    "*.callbacks.*",
]
_LOGGER_EXCLUDE = [
    "*._*",
    "projects.*",
    "imaginaire.*",
]


def _console_filter(record: dict) -> bool:
    from cosmos_framework.utils import log

    # Not sure why but critical messages need a special case to be filtered
    if record["level"].name == "CRITICAL":
        module_name: str = record["name"]
        for pat in _LOGGER_INCLUDE:
            if fnmatch.fnmatch(module_name, pat):
                return True
        for pat in _LOGGER_EXCLUDE:
            if fnmatch.fnmatch(module_name, pat):
                return False
        return True

    if not log._rank0_only_filter(record):
        return False
    module_name: str = record["name"]
    for pat in _LOGGER_INCLUDE:
        if fnmatch.fnmatch(module_name, pat):
            return True
    for pat in _LOGGER_EXCLUDE:
        if fnmatch.fnmatch(module_name, pat):
            return False
    return True


def _init_log_console(*, verbose: bool | None = None):
    from cosmos_framework.utils.flags import VERBOSE
    from cosmos_framework.utils import log

    if verbose is None:
        verbose = VERBOSE

    # Ensure {extra[job_name]} in the logger format always has a value, even when
    # callers (e.g. pytest conftest) bypass _init_script().
    log.logger.configure(extra={"job_name": ""})

    log.logger.remove()
    log.logger.add(
        sys.stdout,
        level="INFO",
        format=_get_logger_format(),
        filter=log._rank0_only_filter if verbose else _console_filter,
        catch=False,
    )
    if not verbose:
        logging.basicConfig(
            level=logging.ERROR,
        )
        loguru.logger.remove()
        warnings.filterwarnings("ignore")


def _init_log_files(output_dir: Path, *, resume: bool = False):
    from cosmos_framework.utils import log

    console_path = output_dir / "console.log"
    debug_path = output_dir / "debug.log"
    log.info(f"Console log saved to {console_path}")
    log.info(f"Debug log saved to {debug_path}")
    logger_format = _get_logger_format()
    file_mode = "a" if resume else "w"
    log.logger.add(
        console_path,
        mode=file_mode,
        level="INFO",
        format=logger_format,
        filter=_console_filter,
        enqueue=True,
        catch=False,
    )
    log.logger.add(
        debug_path,
        mode=file_mode,
        level="DEBUG",
        format=logger_format,
        filter=log._rank0_only_filter,
        enqueue=True,
        catch=False,
    )


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _init_distributed():
    from cosmos_framework.utils import distributed

    distributed.init()


def _cleanup_distributed():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


_error_handler = get_error_handler()


def _distributed_excepthook(exc_type, value, traceback):
    from cosmos_framework.utils import log

    if isinstance(value, Exception):
        _error_handler.record_exception(value)

    log.logger.complete()
    sys.stderr.flush()
    sys.stdout.flush()

    if not is_rank0():
        # Wait for rank0 to throw the exception
        time.sleep(10)

    sys.__excepthook__(exc_type, value, traceback)


def _init_script(training: bool = False, env: dict[str, str] | None = None, default_env: dict[str, str] | None = None):
    """Initialize script."""
    if "imaginaire" in sys.modules:
        raise RuntimeError("'init_script' must be called first.")
    if default_env:
        for k, v in default_env.items():
            os.environ.setdefault(k, v)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if env:
        for k, v in env.items():
            os.environ[k] = v

    _error_handler.initialize()
    sys.excepthook = _distributed_excepthook

    import torch

    if not training:
        torch.set_grad_enabled(False)

    _init_log_console()
    # Initialize distributed early so that:
    # 1. torch.cuda.set_device(local_rank) runs before any CUDA allocations,
    #    ensuring each rank places tensors on its own GPU (not all on cuda:0).
    # 2. sync_model_states in tokenizer / model init is not a silent no-op.
    if enable_distributed():
        _init_distributed()
    set_seed(0)

    if torch.cuda.is_available():
        device_memory_fraction = float(os.environ.get("DEVICE_MEMORY_FRACTION", "1"))
        if device_memory_fraction < 1:
            torch.cuda.set_per_process_memory_fraction(device_memory_fraction)


def _cleanup_script():
    """Clean up script."""
    if sys.exc_info()[1] is not None:
        # Skip cleanup if an exception was raised
        return
    if enable_distributed():
        _cleanup_distributed()


def init_script(
    *, training: bool = False, env: dict[str, str] | None = None, default_env: dict[str, str] | None = None
):
    _init_script(training=training, env=env, default_env=default_env)
    atexit.register(_cleanup_script)


def init_output_dir(output_dir: Path, *, resume: bool = False, job_name: str | None = None):
    """Initialize output directory."""
    from cosmos_framework.utils.flags import FLAGS
    from cosmos_framework.utils import log

    if job_name is not None:
        log.logger.configure(extra={"job_name": job_name})

    output_dir.mkdir(parents=True, exist_ok=True)
    if not is_rank0():
        return

    _init_log_files(output_dir, resume=resume)
    log.debug(f"{FLAGS}")


def set_seed(seed: int):
    """Set seed for random number generator."""
    from transformers import set_seed

    set_seed(seed)
