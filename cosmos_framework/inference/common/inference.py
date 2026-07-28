# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import contextlib
import dataclasses
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ContextManager, Self, Sequence, final

import torch
import torch.profiler

from cosmos_framework.inference.common.args import GuardrailArgs, SampleArgs, SampleOutputs, SetupArgs
from cosmos_framework.inference.common.init import is_rank0
from cosmos_framework.utils import log
from cosmos_framework.utils.misc import TrainingTimer

if TYPE_CHECKING:
    from cosmos_framework.auxiliary.guardrail.common.core import GuardrailRunner


def _download_on_rank0(download: Callable[[], str | Path]) -> Path:
    """Download once and share the resulting path across distributed ranks."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return Path(download())

    payload: list[str | None] = [None, None]
    local_error: Exception | None = None
    if torch.distributed.get_rank() == 0:
        try:
            payload[0] = str(download())
        except Exception as error:
            local_error = error
            payload[1] = f"{type(error).__name__}: {error}"

    torch.distributed.broadcast_object_list(payload, src=0)
    path, error_message = payload
    if error_message is not None:
        if local_error is not None:
            raise local_error
        raise RuntimeError(f"Rank 0 download failed: {error_message}")
    if path is None:
        raise RuntimeError("Rank 0 download returned no path")
    return Path(path)


@contextlib.contextmanager
def sync_distributed_errors():
    """Catches local exceptions and synchronizes the error state across all distributed ranks.

    Raises a DistributedError on all ranks if ANY rank encountered an exception.
    """
    error_flag = torch.zeros(1, dtype=torch.int32, device="cuda")  # [1]
    local_error: Exception | None = None

    try:
        yield
    except Exception as e:
        error_flag += 1
        local_error = e

    if torch.distributed.is_initialized():
        # Sync the error count across all GPUs
        torch.distributed.all_reduce(error_flag, op=torch.distributed.ReduceOp.SUM)

    if error_flag.item() > 0:
        # If we got here, somebody failed.
        # Ranks that failed will raise their actual error.
        # Ranks that succeeded will raise a generic error so they gracefully abort too.
        err_to_raise = local_error if local_error else RuntimeError("A different GPU rank failed.")
        raise err_to_raise


@dataclass
class GuardrailRunners:
    text: "GuardrailRunner"
    video: "GuardrailRunner"

    @classmethod
    def create(cls, args: GuardrailArgs, /) -> Self:
        from cosmos_framework.auxiliary.guardrail.common import presets

        return cls(
            text=presets.create_text_guardrail_runner(offload_model_to_cpu=args.offload_guardrail_models),
            video=presets.create_video_guardrail_runner(offload_model_to_cpu=args.offload_guardrail_models),
        )


@dataclass(kw_only=True)
class Inference(ABC):
    """Inference pipeline base class."""

    setup_args: SetupArgs
    model: torch.nn.Module
    guardrails: GuardrailRunners | None

    _timer: TrainingTimer | None
    _timer_context: list[str] = dataclasses.field(default_factory=list)

    @property
    @abstractmethod
    def model_config(self) -> Any:
        """Get model config."""

    @classmethod
    @abstractmethod
    def _create(cls, setup_args: SetupArgs, /, **kwargs: Any) -> Self:
        """Create instance."""

    @abstractmethod
    def create_batches(
        self, sample_args_list: Sequence[SampleArgs]
    ) -> Iterator[tuple[list[SampleArgs], dict[str, Any]]]:
        """Create batches of sample data."""

    @abstractmethod
    def generate_batch(
        self,
        sample_args_list: Sequence[SampleArgs],
        data_batch: dict[str, Any],
        *,
        save_outputs: bool = True,
    ) -> list[SampleOutputs]:
        """Generate a batch of samples."""

    @final
    @classmethod
    def create(cls, setup_args: SetupArgs, /) -> Self:
        """Create instance."""
        timer = TrainingTimer() if setup_args.benchmark else None
        guardrails = GuardrailRunners.create(setup_args) if setup_args.guardrails else None
        return cls._create(setup_args, guardrails=guardrails, _timer=timer)

    @torch.no_grad()
    @final
    def generate(self, sample_args_list: list[SampleArgs]) -> list[SampleOutputs]:
        """Generate a list of samples."""
        # Create batches
        try:
            with sync_distributed_errors():
                batches = self.create_batches(sample_args_list)
        except Exception as e:
            return [self._handle_sample_exception(sample_args, e) for sample_args in sample_args_list]

        # Generate batches
        sample_outputs: list[SampleOutputs] = []
        for i_batch, (sample_args_batch, data_batch) in enumerate(batches):
            log.debug(f"[{i_batch + 1}] Processing batch", rank0_only=False)

            if self.setup_args.profile:
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                )
            else:
                profiler = contextlib.nullcontext()

            # Spend the *first* warmup pass on a real save pass
            # so the one-time output-persistence cost (dir creation, guardrail
            # lazy-load, first ffmpeg spawn) is paid before measurement begins.
            # Profile the final warmup pass after those one-time costs have been
            # paid. Every measured iteration is then pure generate+decode with
            # uniform timing, and the artifact is produced from the deterministic,
            # identical-seed first warmup pass. When there is no warmup budget,
            # save on the first measured iteration and profile the final one.
            #
            # Note on warmup counts:
            #   * warmup >= 2: the first pass saves, intermediate passes are
            #     throwaway dry-runs, and the final pass is profiled without saving.
            #   * warmup == 1: the sole slot is both the save and profile pass.
            profile = self.setup_args.profile
            save_in_warmup = self.setup_args.warmup > 0
            profile_in_warmup = profile and save_in_warmup
            with self._get_timer_context("warmup"):
                for i_warmup in range(self.setup_args.warmup):
                    is_save_pass = i_warmup == 0
                    is_profile_pass = profile_in_warmup and i_warmup == self.setup_args.warmup - 1
                    profiler_context = profiler if is_profile_pass else contextlib.nullcontext()
                    with self._get_timer(f"{self.__class__.__name__}.generate_batch"), profiler_context:
                        warmup_outputs = self.generate_batch(
                            sample_args_batch,
                            data_batch,
                            save_outputs=is_save_pass,
                        )
                    if is_save_pass:
                        sample_outputs.extend(warmup_outputs)

            num_iterations = self.setup_args.num_iterations
            for i_iteration in range(num_iterations):
                # If the artifact was already saved during warmup, no measured
                # iteration saves; otherwise the first measured iteration does.
                save_outputs = i_iteration == 0 and not save_in_warmup
                if num_iterations > 1:
                    log.debug(
                        f"[{i_batch + 1}] Benchmark iteration {i_iteration + 1}/{num_iterations}",
                        rank0_only=False,
                    )
                # If profiling already ran on the final warmup pass, the measured
                # loop stays profiler-free. Without a warmup, profile the final
                # measured iteration separately from the first-iteration save pass.
                should_profile = i_iteration == num_iterations - 1 and profile and not profile_in_warmup
                profiler_context = profiler if should_profile else contextlib.nullcontext()
                with self._get_timer(f"{self.__class__.__name__}.generate_batch"), profiler_context:
                    batch_outputs = self.generate_batch(
                        sample_args_batch,
                        data_batch,
                        save_outputs=save_outputs,
                    )
                if save_outputs:
                    sample_outputs.extend(batch_outputs)

            if self.setup_args.profile and is_rank0():
                assert isinstance(profiler, torch.profiler.profile)
                sample_args = sample_args_batch[0]
                profile_file = sample_args.output_dir / "profile.json.gz"
                profiler.export_chrome_trace(str(profile_file))
                log.success(f"Saved profile to '{profile_file}'")

        return sample_outputs

    def _get_timer(self, func_name: str) -> ContextManager:
        if self._timer is None:
            return nullcontext()
        if self._timer_context:
            context = ".".join(self._timer_context)
            func_name = f"[{context}] {func_name}"
        return self._timer(func_name)

    @contextmanager
    def _get_timer_context(self, func_name: str):
        self._timer_context.append(func_name)
        try:
            yield
        finally:
            self._timer_context.pop()

    def get_timer_results(self) -> dict | None:
        if self._timer is None:
            return None
        # With warmup == 0 and multiple measured iterations, the first
        # measured iteration is the cold pass and is dropped from the
        # reported average.
        drop_first_non_warmup = self.setup_args.benchmark and self.setup_args.warmup == 0
        average: dict[str, float] = {}
        for key, values in self._timer.results.items():
            if not values:
                continue
            if drop_first_non_warmup and not key.startswith("[warmup]") and len(values) > 1:
                values = values[1:]
            average[key] = sum(values) / len(values)
        return {
            "all": self._timer.results,
            "average": average,
        }

    def _handle_sample_exception(self, sample_args: SampleArgs, e: Exception) -> SampleOutputs:
        msg = f"Error generating sample '{sample_args.name}': {e}"
        if not self.setup_args.keep_going:
            raise ValueError(msg) from e
        log.error(msg)
        return SampleOutputs(
            args=sample_args.model_dump(mode="json"), status="error", message=msg, stack_trace=traceback.format_exc()
        )

    @final
    def _run_text_guardrail(self, name: str, prompt: str) -> None:
        """Run guardrail checks on the prompt."""
        if self.guardrails is None:
            return

        from cosmos_framework.auxiliary.guardrail.common import presets

        if not presets.run_text_guardrail(prompt, self.guardrails.text):
            raise ValueError(f"Guardrail blocked prompt '{name}': '{prompt}'")

    @final
    def _run_video_guardrail(self, name: str, video_cthw: torch.Tensor) -> torch.Tensor:
        """Run guardrail checks on the video and apply face blur."""
        if self.guardrails is None:
            return video_cthw
        processed_video_cthw, message = _run_video_guardrail(self.guardrails.video, video_cthw)
        if processed_video_cthw is None:
            raise ValueError(f"Guardrail blocked video '{name}': {message}")
        return processed_video_cthw


def _run_video_guardrail(
    video_guardrail_runner: "GuardrailRunner", video_cthw: torch.Tensor
) -> tuple[torch.Tensor | None, str]:
    """Run video guardrail and apply face blur.

    Returns a ``(video_or_none, message)`` tuple. When the guardrail blocks
    the video, ``video_or_none`` is ``None`` and ``message`` contains the
    underlying reason (unsafe frame ratio, categories, etc.) as produced by
    :class:`GuardrailRunner.run_safety_check`.
    """
    if video_cthw.ndim != 4:
        raise ValueError(f"Video tensor must have 4 dimensions, got {video_cthw.shape}")
    frames_thwc = (
        (video_cthw * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 2, 3, 0).detach().cpu().numpy()
    )  # [T,H,W,C]

    # Inline of presets.run_video_guardrail so we can forward `message` (the helper drops it).
    is_safe, message = video_guardrail_runner.run_safety_check(frames_thwc)
    if not is_safe:
        log.critical(f"GUARDRAIL BLOCKED: {message}")
        return None, message

    frames_thwc = video_guardrail_runner.postprocess(frames_thwc)
    video_cthw = (torch.from_numpy(frames_thwc).float().permute(3, 0, 1, 2) / 255.0).to(  # [C,T,H,W]
        video_cthw.device, dtype=video_cthw.dtype
    )
    return video_cthw, message
