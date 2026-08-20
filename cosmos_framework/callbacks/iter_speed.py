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
            return
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

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

        tokens_suffix = (
            f" | {token_stats['tokens_per_iteration']:,.0f} tokens per iteration"
            f" ({token_stats.get('tokens_per_second', 0.0):,.0f} tokens/s)"
            if token_stats
            else ""
        )
        log.info(
            f"{iteration} : iter_speed {iter_speed:.2f} seconds per iteration | "
            f"Loss: {loss.detach().item():.4f}{tokens_suffix}",
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
