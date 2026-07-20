# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Distillation-tailored DCP for Cosmos3.

Extends the Cosmos3 VFM DistributedCheckpointer to support multi-model /
multi-optimizer training (e.g., student + fake-score + discriminator).
"""

import functools
import os
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import log, misc
from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.checkpoint.dcp import (
    AsyncMode,
    CustomLoadPlanner,
    _DataloaderWrapper,
)
from cosmos_framework.checkpoint.dcp import (
    DistributedCheckpointer as _DistributedCheckpointer,
)
from cosmos_framework.checkpoint.dcp import ModelWrapper as VFMModelWrapper
from cosmos_framework.utils.generator.rand_state import get_rand_state_dict, set_rand_state_dict
from cosmos_framework.model.generator.distillation.optimizer import OptimizerContainerLike, is_optimizer_container

__all__: tuple[str, ...] = (
    "DistributedCheckpointer",
    "ModelWrapper",
    "OptimizerWrapper",
)


class OptimizerWrapper(Stateful):
    """FSDP/DCP-aware optimizer state wrapper for per-phase distillation optimizers."""

    def __init__(
        self,
        model: torch.nn.Module | list[torch.nn.Module],
        optim: torch.optim.Optimizer | OptimizerContainerLike | list[torch.optim.Optimizer],
    ) -> None:
        self.optim_container: OptimizerContainerLike | None = optim if is_optimizer_container(optim) else None
        if self.optim_container is not None:
            self.model: list[torch.nn.Module] = []
            self.optim: list[torch.optim.Optimizer] = []
            return

        self.model = [model] if isinstance(model, torch.nn.Module) else model
        self.optim = [optim] if isinstance(optim, torch.optim.Optimizer) else optim
        if len(self.model) != len(self.optim):
            raise ValueError(f"Expected matched model/optimizer lists, got {len(self.model)} and {len(self.optim)}")

    def state_dict(self) -> dict[str, Any]:
        if self.optim_container is not None:
            return self.optim_container.state_dict()

        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {k: v for sd in map(func, self.model, self.optim) for k, v in sd.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.optim_container is not None:
            self.optim_container.load_state_dict(state_dict)
            return

        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model, self.optim))


class ModelWrapper(VFMModelWrapper):
    """
    Modify the Cosmos3 VFM model wrapper to exclude
    the teacher weights from the model state dict.
    """

    def __init__(
        self,
        model: ImaginaireModel,
        exclude_teacher_weights: bool = True,
        strict_resume: bool = True,
    ) -> None:
        super().__init__(model)
        self.exclude_teacher_weights: bool = exclude_teacher_weights
        self.strict_resume: bool = strict_resume

    def _unregistered_fake_score(self) -> torch.nn.Module | None:
        """Return ``net_fake_score`` only when it is held outside the module registry.

        Self-forcing DMD and dense DMD2 hide ``net_fake_score`` from the module
        registry (so inference ``get_model_state_dict`` stays student-only); it
        must therefore be serialized explicitly. Older layouts may have registered
        ``net_fake_score``, so those must NOT be double-handled.
        """
        if "net_fake_score" in dict(self.model.named_children()):
            return None
        fake_score = getattr(self.model, "net_fake_score", None)
        return fake_score if isinstance(fake_score, torch.nn.Module) else None

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        if self.exclude_teacher_weights:
            sd = {k: v for k, v in sd.items() if not k.startswith("net_teacher.")}
        fake_score = self._unregistered_fake_score()
        if fake_score is not None:
            for k, v in get_model_state_dict(fake_score).items():
                sd[f"net_fake_score.{k}"] = v
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> _IncompatibleKeys:
        # DCP/FSDP-aware load path
        fake_score = self._unregistered_fake_score()
        fake_score_results: _IncompatibleKeys | None = None
        main_state_dict = state_dict
        if fake_score is not None:
            prefix = "net_fake_score."
            fake_score_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            main_state_dict = {k: v for k, v in state_dict.items() if not k.startswith(prefix)}
            fake_score_results = set_model_state_dict(
                model=fake_score,
                model_state_dict=fake_score_state_dict,
                options=StateDictOptions(strict=False, ignore_frozen_params=False),
            )

        results = set_model_state_dict(
            model=self.model,
            model_state_dict=main_state_dict,
            options=StateDictOptions(strict=False, ignore_frozen_params=False),
        )

        if self.strict_resume:
            bad_missing = [k for k in results.missing_keys if not k.startswith("net_teacher.")]
            bad_unexpected = [k for k in results.unexpected_keys if not k.startswith("net_teacher.")]
            if fake_score_results is not None:
                bad_missing += list(fake_score_results.missing_keys)
                bad_unexpected += list(fake_score_results.unexpected_keys)
            if bad_missing or bad_unexpected:
                raise ValueError(
                    f"Strict resume failed. Missing(non-teacher)={bad_missing[:20]}, "
                    f"Unexpected(non-teacher)={bad_unexpected[:20]}"
                )
        return results


class DistributedCheckpointer(_DistributedCheckpointer):
    @misc.timer("checkpoint loading")
    def load(
        self,
        model: ImaginaireModel,
        optimizer: Any = None,
        scheduler: Any = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_start(model)

        model_dict = model.model_dict()

        resume_keys, checkpoint_path, warm_start = self.keys_to_resume_during_load()
        resume_keys = sorted(resume_keys)
        log.critical(f"Resuming ckpt {checkpoint_path} with keys: {resume_keys}")

        iteration = 0

        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            for key in resume_keys:
                dist.barrier()

                cur_key_ckpt_full_path = os.path.join(checkpoint_path, key)
                log.critical(f"Start loading checkpoint from {cur_key_ckpt_full_path}")

                strict_resume = self.config_checkpoint.strict_resume
                keys_to_skip_loading = self.config_checkpoint.keys_to_skip_loading if warm_start else []
                load_planner = CustomLoadPlanner(
                    allow_partial_load=not strict_resume,
                    keys_to_skip_loading=keys_to_skip_loading,
                )

                if key == "model":
                    storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                    log.info("- Loading the model...")
                    _model_wrapper = ModelWrapper(
                        model,
                        exclude_teacher_weights=True,
                        strict_resume=strict_resume,
                    )
                    _state_dict = _model_wrapper.state_dict()
                    dcp.load(_state_dict, storage_reader=storage_reader, planner=load_planner)
                    _model_wrapper.load_state_dict(_state_dict)

                elif key == "optim":
                    if optimizer is None:
                        raise ValueError("optimizer must be provided when loading optim state.")
                    for optim_key, optim in optimizer.items():
                        storage_reader = self.get_storage_reader(f"{cur_key_ckpt_full_path}_{optim_key}")
                        log.info(f"- Loading the optimizer ({optim_key})...")
                        _optim_wrapper = OptimizerWrapper(model_dict[optim_key], optim)
                        _state_dict = _optim_wrapper.state_dict()
                        optim_load_planner = CustomLoadPlanner(
                            allow_partial_load=not strict_resume,
                            keys_to_skip_loading=[],
                        )
                        dcp.load(
                            _state_dict,
                            storage_reader=storage_reader,
                            planner=optim_load_planner,
                        )
                        _optim_wrapper.load_state_dict(_state_dict)

                elif key == "scheduler":
                    if scheduler is None:
                        raise ValueError("scheduler must be provided when loading scheduler state.")
                    for sched_key, sched in scheduler.items():
                        storage_reader = self.get_storage_reader(f"{cur_key_ckpt_full_path}_{sched_key}")
                        log.info(f"- Loading the scheduler ({sched_key})...")
                        _state_dict = sched.state_dict()
                        dcp.load(
                            _state_dict,
                            storage_reader=storage_reader,
                            planner=load_planner,
                        )
                        sched.load_state_dict(_state_dict)

                elif key == "trainer":
                    if grad_scaler is None:
                        raise ValueError("grad_scaler must be provided when loading trainer state.")
                    storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                    log.info("- Loading the trainer...")

                    rng_key = f"rng_state_{dist.get_rank()}"
                    current_rng_state = get_rand_state_dict()
                    _state_dict = {
                        "grad_scaler": grad_scaler.state_dict(),
                        "iteration": iteration,
                    }
                    metadata = storage_reader.read_metadata()
                    rng_key_exists = any(
                        k.startswith(f"{rng_key}.") or k == rng_key for k in metadata.state_dict_metadata.keys()
                    )
                    if rng_key_exists:
                        _state_dict[rng_key] = current_rng_state

                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    grad_scaler.load_state_dict(_state_dict["grad_scaler"])
                    iteration = _state_dict["iteration"]
                    set_rand_state_dict(_state_dict.get(rng_key, current_rng_state))

                elif key == "dataloader":
                    # Per-rank dataloader-state resume (main's base checkpointer). Skips
                    # gracefully when no dataloader-state callback is registered or no
                    # dataloader checkpoint was written.
                    if not easy_io.exists(cur_key_ckpt_full_path, backend_key=self.load_s3_backend_key):
                        log.info(
                            f"Checkpoint {cur_key_ckpt_full_path} does not exist, skip loading dataloader.",
                            rank0_only=False,
                        )
                        continue
                    rank = dist.get_rank()
                    dataloader_pkl_path = os.path.join(cur_key_ckpt_full_path, f"rank_{rank}.pkl")
                    if not easy_io.exists(dataloader_pkl_path, backend_key=self.load_s3_backend_key):
                        log.info(f"No dataloader checkpoint found at {dataloader_pkl_path}", rank0_only=False)
                        continue
                    log.info(f"- Loading the dataloader {cur_key_ckpt_full_path}...", rank0_only=False)
                    _state_dict = easy_io.load(
                        dataloader_pkl_path,
                        file_format="pkl",
                        backend_key=self.load_s3_backend_key,
                    )
                    dataloader_wrapper = _DataloaderWrapper(self.callbacks)
                    if dataloader_wrapper.has_state():
                        dataloader_wrapper.load_state_dict(_state_dict)

                else:
                    raise ValueError(f"Invalid key: {key}. not support to resume.")

            if self.callbacks is not None:
                self.callbacks.on_load_checkpoint(model, state_dict=_state_dict)
            log.info(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")
        else:
            log.info("Training from scratch.")

        torch.cuda.empty_cache()

        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=checkpoint_path)
        return iteration

    # No custom save_state_dict_worker needed: the save() method below
    # pre-flattens optim/scheduler into per-key top-level entries so the
    # parent's save_state_dict_worker (used by both sync and async paths)
    # writes each to its own subdirectory (e.g. optim_net/, optim_fake_score/).

    def save(
        self,
        model: ImaginaireModel,
        optimizer: Any,
        scheduler: Any = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        iteration: int = 0,
    ) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._wait_for_previous_async_checkpoint()

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_start(model, iteration)

        model_dict = model.model_dict()
        checkpoint_file = f"iter_{iteration:09}"

        rng_key = f"rng_state_{dist.get_rank()}"
        to_save_dict: dict[str, Any] = {
            "model": ModelWrapper(model, exclude_teacher_weights=True).state_dict(),
            "trainer": {
                "grad_scaler": grad_scaler.state_dict(),
                "iteration": iteration,
                rng_key: get_rand_state_dict(),
            },
        }
        # Flatten optimizer and scheduler dicts into separate top-level keys
        # (e.g. "optim_net", "optim_fake_score", "scheduler_net", …) so that
        # the parent's save_state_dict_worker — which is used by the async
        # background process — creates the correct per-key subdirectories.
        for optim_key, optim in optimizer.items():
            to_save_dict[f"optim_{optim_key}"] = OptimizerWrapper(model_dict[optim_key], optim).state_dict()
        for sched_key, sched in scheduler.items():
            to_save_dict[f"scheduler_{sched_key}"] = sched.state_dict()

        # Per-rank dataloader state (main's base checkpointer). No-op unless a
        # callback tagged ``checkpoint_component=="dataloader"`` is registered;
        # the inherited save_state_dict_worker writes the "dataloader" key as pkl.
        dataloader_wrapper = _DataloaderWrapper(self.callbacks)
        if dataloader_wrapper.has_state():
            to_save_dict["dataloader"] = dataloader_wrapper.state_dict()

        for key in list(to_save_dict.keys()):
            output_dirname = os.path.join(self.save_dirname, f"iter_{iteration:09}/{key}")
            to_save_dict[key] = (to_save_dict[key], output_dirname)

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint(model, state_dict=to_save_dict)

        log.info(f"Saving checkpoint to {os.path.join(self.save_dirname, checkpoint_file)}")

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            # PhaseOptimizer state is structurally different before and after a
            # phase's first optimizer step. Recreate the pinned staging tree for
            # each distillation save so DCP does not try to copy new optimizer
            # slots into a stale CPU companion structure.
            self.cpu_offload_state_dict = None
            self._checkpoint_async_with_pinned_memory(checkpoint_file, to_save_dict)
        else:
            start_time = time.monotonic()
            try:
                self.save_state_dict_worker(to_save_dict, checkpoint_file)
            finally:
                if self.callbacks is not None:
                    self.callbacks.on_save_checkpoint_success(
                        iteration=iteration, elapsed_time=time.monotonic() - start_time
                    )

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)
