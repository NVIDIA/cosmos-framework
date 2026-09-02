# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import time

import torch
import wandb
from torch import Tensor

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils import log
from cosmos_framework.utils.distributed import is_rank0
from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.utils.generator.method_timer import MethodTimer

# Packed length of this rank's batch, written by ``_PackingMetrics.attach_to`` in
# cosmos_framework.data.generator.joint_dataloader. Absent for loaders that do not
# pack, in which case the token metrics are skipped.
_NUM_TOKENS_KEY = "_num_tokens"


class IterSpeed(EveryN):
    """Log training throughput: seconds per iteration, and the tokens retired in them.

    Both token metrics are totals over all ranks and averages over the logging window,
    so ``tokens_per_iteration`` divided by ``iter_speed`` is ``tokens_per_second``. They
    are omitted for loaders that do not report a packed length, and are corrected for
    the batch that context-parallel ranks share rather than each fetching their own.

    Args:
        hit_thres (int): Number of iterations to wait before logging.
        save_s3 (bool): Whether to save to S3.
        save_s3_every_log_n (int): Save to S3 every n log iterations, which means save_s3_every_log_n n * every_n global iterations.
    """

    def __init__(self, *args, hit_thres: int = 5, save_s3: bool = True, save_s3_every_log_n: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = None
        self.hit_counter = 0
        self.hit_thres = hit_thres
        self.save_s3 = save_s3
        self.save_s3_every_log_n = save_s3_every_log_n
        self.name = self.__class__.__name__
        self.last_hit_time = time.time()
        # Summed over ranks only when logging, so per-step accounting costs no collective.
        self._local_tokens_since_log = 0
        # Installed on ``on_train_start`` for models that expose ``.encode`` (the VFM/VAE
        # path); stays None for models without a VAE (e.g. the VLM path), so the derived
        # metrics below are simply omitted rather than reported as zero. Read out once per
        # training step in ``_drain_step_timers``, not once per logging window: a step-level
        # reading needs no explicit synchronize (see that method), while draining a whole
        # window's worth of events at once would.
        self._vae_timer: MethodTimer | None = None
        # Same convention as ``_vae_timer``, timing ``OmniMoTModel._prepare_training_data``
        # (tokenize + VAE encode) instead of the VAE encode alone.
        self._prepare_data_timer: MethodTimer | None = None
        # This rank's own per-step seconds since the last logging window, one entry per
        # step, oldest first. Reset to `[]` every window in ``every_n_impl``.
        self._vae_sec_per_step: list[float] = []
        self._prepare_data_sec_per_step: list[float] = []

    def _restore_timers(self) -> None:
        """Undo both monkeypatches and drop the timers. Idempotent.

        Called from both ends of the training lifecycle: ``on_train_end`` for the normal
        path, and the top of ``on_train_start`` so that a second start -- a harness that
        re-runs the callback, or any future caller that enters ``train()`` twice in one
        process -- reinstalls onto the model's own method rather than onto the previous
        timer's wrapper. ``MethodTimer.__init__`` captures ``getattr(model, method_name)``
        as the method to call, so without this each start would capture the prior
        ``_timed_call`` and nest: every layer stays on the call path forever, adding a
        Python-level dispatch per call, and the outer timer's events would bracket the
        inner wrappers' work along with the method's.
        """
        for timer in (self._vae_timer, self._prepare_data_timer):
            if timer is not None:
                timer.restore()
        self._vae_timer = None
        self._prepare_data_timer = None
        # Readings belonging to a finished run; a fresh start must not report them, and
        # ``_reduce_step_series`` requires the series length to match across ranks.
        self._vae_sec_per_step = []
        self._prepare_data_sec_per_step = []

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        self._restore_timers()
        # ``every_n`` of 0 (or unset) means ``EveryN`` never dispatches ``every_n_impl`` from a
        # training step, so nothing would ever drain these timers' per-step buffers and they
        # would grow for the whole run. Skip installing any of them rather than leak.
        if not self.every_n:
            return
        if hasattr(model, "encode"):
            self._vae_timer = MethodTimer(model, "encode")
            self._vae_timer.reset()
        if hasattr(model, "_prepare_training_data"):
            self._prepare_data_timer = MethodTimer(model, "_prepare_training_data")
            self._prepare_data_timer.reset()

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        # Leaves the model exactly as it was found. Post-training work that calls
        # ``encode`` (final validation, a sampling callback, an export path) then runs
        # against the real method instead of through a timer nothing will ever read.
        self._restore_timers()

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.hit_counter < self.hit_thres:
            log.info(
                f"Iteration {iteration}: "
                f"Hit counter: {self.hit_counter + 1}/{self.hit_thres} | "
                f"Loss: {loss.detach().item():.4f} | "
                f"Time: {time.time() - self.last_hit_time:.2f}s",
            )
            self.hit_counter += 1
            self.last_hit_time = time.time()
            #! useful for large scale training and avoid oom crash in the first two iterations!!!
            torch.cuda.synchronize()
            self._drain_step_timers(record=False)
            return
        self._drain_step_timers(record=True)
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    def _drain_step_timers(self, *, record: bool) -> None:
        """Read this step's elapsed time off each installed timer and start its next window.

        Called every step, warm-up included: a timer left running across the warm-up
        boundary would fold one-time compile cost into the first real step's reading.
        ``record=False`` reads the value only to reset the timer, then throws it away.

        No ``torch.cuda.synchronize()`` here, deliberately: this rank's own preceding
        forward/backward/optimizer-step work is expected to have already completed on the
        GPU by the time a training step ends and this callback fires -- typically because
        something else in the step already forced a device sync (a loss or grad-norm
        ``.item()``, most likely). ``Event.elapsed_time`` (inside ``stop_and_read_ms``) does
        not silently return a wrong number if that expectation is ever false: the CUDA
        runtime raises ``cudaErrorNotReady`` for an event that has not completed, which
        surfaces here as a ``RuntimeError`` rather than a bad reading.
        """
        if self._vae_timer is not None:
            sec = self._vae_timer.stop_and_read_ms() / 1000.0
            self._vae_timer.reset()
            if record:
                self._vae_sec_per_step.append(sec)
        if self._prepare_data_timer is not None:
            sec = self._prepare_data_timer.stop_and_read_ms() / 1000.0
            self._prepare_data_timer.reset()
            if record:
                self._prepare_data_sec_per_step.append(sec)

    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        """Tally this micro-batch's tokens.

        Every fetched batch lands here, whereas ``on_training_step_end`` fires once per
        optimizer step and so would see only the last micro-batch of a gradient
        accumulation window. Accumulating is local and costs no collective.
        """
        self._local_tokens_since_log += int(data_batch.get(_NUM_TOKENS_KEY, 0))

    @staticmethod
    def _cp_size(model: ImaginaireModel) -> int:
        """The context-parallel degree, which is how many times the ranks report one batch.

        CP ranks do not each train on their own data. Every rank fetches one batch per
        ``cp_size``-step window and ``ImaginaireTrainer._fetch_data_batch`` replays that
        cached batch for every slot in the window, while the model trains on one slot
        owner's batch at a time. Each rank's tokens therefore appear ``cp_size`` times
        across a window, and the group really retires one rank's batch per step, so
        dividing by ``cp_size`` is exact over whole windows.
        """
        parallel_dims = getattr(model, "parallel_dims", None)
        if parallel_dims is None or not getattr(parallel_dims, "cp_enabled", False):
            return 1
        return max(1, int(parallel_dims.cp_size))

    @staticmethod
    def _reduce_step_series(values: list[float]) -> tuple[float, float] | None:
        """Return (avg of the per-step cross-rank max, global avg over ranks and steps).

        Two ``all_reduce`` collectives (SUM, then MAX) rather than an ``all_gather``: both
        produce a result the same size as one rank's own input regardless of world_size,
        unlike gathering every rank's series onto every rank, which would cost O(world_size)
        on every rank -- worth avoiding at the node counts this trains at.

        The max captures imbalance the way the old barrier-based ``sync`` measurement did,
        without a barrier: for each step, the slowest rank's reading IS the cost the whole
        step actually paid, since every other rank finishes at or before it and the next
        collective (the gradient all-reduce) waits for the slowest one regardless. Averaging
        that max over the window's steps gives one number for "how expensive did the worst
        rank's share of this get", while the global average gives the plain per-rank cost.

        ``values`` is this rank's own per-step series, one entry per training step since the
        last window read, and must be the same length on every rank: every rank calls
        ``on_training_step_end`` (and therefore ``_drain_step_timers``) the same number of
        times per window, in lockstep, and this timer is installed identically on every rank
        or not at all (the ``hasattr`` checks in ``on_train_start`` depend only on the model
        class, which is homogeneous across ranks). Runs on every rank unconditionally, even
        with this rank's own list empty (its timer was never installed at all, e.g. no VAE
        path): a rank that skipped the collective would hang every other rank waiting on it.

        ``None`` if the series is empty -- e.g. this timer was never installed on any rank,
        or this is a window with no recorded steps.
        """
        n = len(values)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local = torch.tensor(values, dtype=torch.float64, device=device)

        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            if n == 0:
                return None
            avg = local.mean().item()
            return avg, avg

        world_size = torch.distributed.get_world_size()
        summed = local.clone()
        torch.distributed.all_reduce(summed, op=torch.distributed.ReduceOp.SUM)
        maxed = local.clone()
        torch.distributed.all_reduce(maxed, op=torch.distributed.ReduceOp.MAX)

        if n == 0:
            return None
        avg_of_max = maxed.mean().item()
        global_avg = (summed.sum() / (world_size * n)).item()
        return avg_of_max, global_avg

    @staticmethod
    def _summary_stats(name: str, summary: tuple[float, float] | None, iter_speed: float) -> dict[str, float]:
        """Expand one timer's ``(avg_of_max, global_avg)`` into the metrics logged for it.

        ``avg`` is the plain per-rank-per-step mean; ``max`` is the average, over the
        window's steps, of that step's slowest rank -- see ``_reduce_step_series`` for why
        that stands in for the old barrier-measured wait time without a barrier. Both
        already ARE per-iteration seconds (one entry per step), so neither is divided by
        ``iterations_per_window`` the way ``token_stats`` is. Already reduced across ranks
        by the caller, on every rank -- just unpacked here.

        Every timer goes through this one function so the avg/max pair is always built the
        same way. Each fraction is divided by ``iter_speed`` from the SAME aggregation as
        its numerator: ``_frac`` from the global average, ``_frac_max`` from the per-step
        max. Timers previously expanded their own summaries inline and had drifted apart --
        one reported only an average-based fraction and the other only a max-based one,
        under names similar enough to read as comparable in W&B when they were not.

        ``{}`` when the timer was never installed, which keeps it out of the logs entirely
        rather than reporting it as zero.
        """
        if summary is None:
            return {}
        avg_of_max, global_avg = summary
        stats = {
            f"{name}_sec_per_iteration": global_avg,
            f"{name}_sec_per_iteration_max": avg_of_max,
        }
        if iter_speed > 0:
            stats[f"{name}_frac"] = global_avg / iter_speed
            stats[f"{name}_frac_max"] = avg_of_max / iter_speed
        return stats

    @staticmethod
    def _summary_suffix(name: str, stats: dict[str, float]) -> str:
        """One timer's contribution to the per-window log line, or ``""`` if uninstalled.

        Each percentage sits beside the number it was computed from, so neither can be
        read as belonging to the other.
        """
        if not stats:
            return ""
        return (
            f" | {name} {stats[f'{name}_sec_per_iteration']:.3f}s/iter avg"
            f" ({stats.get(f'{name}_frac', 0.0):.1%})"
            f", max {stats[f'{name}_sec_per_iteration_max']:.3f}s"
            f" ({stats.get(f'{name}_frac_max', 0.0):.1%})"
        )

    def _reduce_tokens_since_log(self) -> int:
        """Tokens reported across every rank since the last log, resetting this rank's tally.

        Each rank packs its own batch, so a total needs a sum over the world; the CP
        double counting that sum inherits is undone by the caller. Runs on every rank
        rather than under ``rank0_only``, which would hang, and runs unconditionally, so
        no rank can be waiting in a collective the others skipped.
        """
        tokens = self._local_tokens_since_log
        self._local_tokens_since_log = 0
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return tokens
        device = "cuda" if torch.cuda.is_available() else "cpu"
        total = torch.tensor([tokens], dtype=torch.int64, device=device)
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        return int(total.item())

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, Tensor],
        output_batch: dict[str, Tensor],
        loss: Tensor,
        iteration: int,
    ) -> None:
        # Before the early return, so the window's tokens line up with the window its
        # elapsed time measures, and so every rank reaches the collective.
        tokens_this_window = self._reduce_tokens_since_log() // self._cp_size(model)

        # Same placement, same reasoning: these are collectives (two all_reduce calls each)
        # and must run on every rank every window, regardless of the early return below.
        # Unlike the old per-window vae/prepare-data read, nothing here needs a
        # ``torch.cuda.synchronize()`` first -- ``_drain_step_timers`` already read each
        # step's value out (and reset the timer for the next one) back when that step ended,
        # so by now these are plain Python floats with no outstanding GPU work behind them.
        # The reduction is already identical on every rank once this returns, so rank0 can
        # just read the result back out later rather than reducing again.
        vae_summary = self._reduce_step_series(self._vae_sec_per_step)
        prepare_data_summary = self._reduce_step_series(self._prepare_data_sec_per_step)
        self._vae_sec_per_step = []
        self._prepare_data_sec_per_step = []

        if self.time is None:
            self.time = time.time()
            return
        # EveryN dispatches here only after dividing by every_n itself, so it cannot be
        # None; bind it once to say so rather than repeating the assumption downstream.
        assert self.every_n is not None, "EveryN dispatched every_n_impl with every_n unset."
        iterations_per_window = self.every_n * self.step_size
        cur_time = time.time()
        iter_speed = (cur_time - self.time) / iterations_per_window

        # Averaged over the same window as iter_speed, so tokens_per_second is the
        # window's throughput rather than one sampled batch scaled by an average time.
        token_stats: dict[str, float] = {}
        if tokens_this_window > 0:
            token_stats["tokens_per_iteration"] = tokens_this_window / iterations_per_window
            if iter_speed > 0:
                token_stats["tokens_per_second"] = token_stats["tokens_per_iteration"] / iter_speed

        if not is_rank0():
            self.time = cur_time
            return

        vae_stats = self._summary_stats("vae_encode", vae_summary, iter_speed)
        prepare_data_stats = self._summary_stats("prepare_data", prepare_data_summary, iter_speed)

        tokens_suffix = (
            f" | {token_stats['tokens_per_iteration']:,.0f} tokens per iteration"
            f" ({token_stats.get('tokens_per_second', 0.0):,.0f} tokens/s)"
            if token_stats
            else ""
        )
        vae_suffix = self._summary_suffix("vae_encode", vae_stats)
        prepare_data_suffix = self._summary_suffix("prepare_data", prepare_data_stats)
        log.info(
            f"{iteration} : iter_speed {iter_speed:.2f} seconds per iteration | "
            f"Loss: {loss.detach().item():.4f}"
            f"{tokens_suffix}{vae_suffix}{prepare_data_suffix}",
        )

        per_sample_batch_counter = dict()
        # for VFM
        if hasattr(model, "is_image_batch") and hasattr(model, "input_image_key") and hasattr(model, "input_video_key"):
            is_image_batch = model.is_image_batch(data_batch)
            if is_image_batch:
                image_batch_size = len(data_batch[model.input_image_key])
                per_sample_batch_counter["image_batch_size"] = image_batch_size
            else:
                video_batch_size = len(data_batch[model.input_video_key])
                per_sample_batch_counter["video_batch_size"] = video_batch_size
        # for LLM training only
        elif "input_ids" in data_batch:
            mbs = data_batch["input_ids"].shape[0]
            dp_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            grad_accum_iter = int(trainer.config.trainer.grad_accum_iter)
            per_sample_batch_counter["token_batch_size"] = mbs
            per_sample_batch_counter["token_global_batch_size"] = mbs * dp_size * grad_accum_iter
            # Cumulative token count (LLM analog of sample_counter). Set by
            # ``LLMPretrainModel.training_step`` into a persistent buffer on
            # ``model.net``, so this value survives checkpoint resume.
            if hasattr(model, "token_counter"):
                per_sample_batch_counter["token_counter"] = model.token_counter

        if wandb.run:
            sample_counter = getattr(trainer, "sample_counter", iteration)
            wandb.log(
                {
                    "timer/iter_speed": iter_speed,
                    "sample_counter": sample_counter,
                }
                | {f"timer/{name}": value for name, value in token_stats.items()}
                | {f"timer/{name}": value for name, value in vae_stats.items()}
                | {f"timer/{name}": value for name, value in prepare_data_stats.items()}
                | per_sample_batch_counter,
                step=iteration,
            )
        self.time = cur_time
        if self.save_s3:
            if iteration % (self.save_s3_every_log_n * self.every_n) == 0:
                easy_io.dump(
                    {
                        "iter_speed": iter_speed,
                        "iteration": iteration,
                    }
                    | token_stats,
                    f"s3://rundir/{self.name}/iter_{iteration:09d}.yaml",
                )
