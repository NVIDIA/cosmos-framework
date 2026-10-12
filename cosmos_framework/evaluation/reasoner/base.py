# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base evaluator for Cosmos Framework."""

from __future__ import annotations

import logging as log
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cosmos_framework.checkpoint.reasoner import distributed_identity
from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime
from cosmos_framework.utils.workflow_status import log_workflow_status

COMPONENT_NAME = "Cosmos Framework Evaluation"


class BaseEvaluator(ABC):
    """Base evaluator for Cosmos Framework."""

    def __init__(self, config: Dict[str, Any], enable_lora: bool = False) -> None:
        """Initialize the BaseEvaluator."""
        self.config = config
        self.enable_lora = enable_lora
        self.model = None
        self.processor = None
        self.model_config = config.get("model", {})
        self.eval_config = config.get("evaluation", {})
        self.gen_config = config.get("generation", {})
        self.vision_config = dict(config.get("vision", {}))
        # Accept both the public Cosmos spelling and qwen-vl-utils' legacy alias,
        # but keep num_frames canonical for the Framework runtime.
        num_frames = self.vision_config.get("num_frames")
        qwen_nframes = self.vision_config.get("nframes")
        if num_frames is not None and qwen_nframes is not None:
            if int(num_frames) != int(qwen_nframes):
                raise ValueError("vision.num_frames and vision.nframes must match when both are set")
        configured_frames = num_frames if num_frames is not None else qwen_nframes
        if configured_frames is not None:
            configured_frames = int(configured_frames)
            if configured_frames <= 0:
                raise ValueError("vision.num_frames must be positive")
            self.vision_config["num_frames"] = configured_frames
            self.vision_config["nframes"] = configured_frames
        self.dataset_cfg = config.get("dataset", {})
        self.video_preprocessor = None

    def _send_status_callback(self, message: str) -> None:
        """Send status callback to prevent timeout.

        Args:
            message: Status message to send
        """
        log_workflow_status({"progress": message}, component_name=COMPONENT_NAME)

    def load_model(self) -> Tuple[Any, Any]:
        """Load the model and processor with common logic."""
        log.info("Loading model and processor...")
        start_time = time.time()

        decoder_name = self.vision_config.get("video_decoder", "pynvvideocodec")
        if decoder_name == "pynvvideocodec":
            from cosmos_framework.inference.reasoner.pynv_video_reader import register_pynv_video_reader

            decoder = register_pynv_video_reader(
                cache_size=int(self.vision_config.get("video_cache_size", 0)),
                video_override_map=self.vision_config.get("video_override_map"),
                strict=True,
            )
            log.info("GPU video decoder registered: %s", decoder)
        elif decoder_name == "torchcodec-cuda-on-demand":
            from cosmos_framework.inference.reasoner.framework_torchcodec_video import (
                FrameworkTorchCodecVideoPreprocessor,
            )

            self.video_preprocessor = FrameworkTorchCodecVideoPreprocessor(
                num_frames=int(self.vision_config["num_frames"]),
                cache_size=int(self.vision_config.get("video_cache_size", 0)),
                process_threads=int(self.vision_config.get("process_threads", 8)),
                decoder_threads=int(self.vision_config.get("decoder_threads", 1)),
                dataloader_num_workers=int(self.vision_config.get("dataloader_num_workers", 1)),
                dataloader_prefetch_factor=int(self.vision_config.get("dataloader_prefetch_factor", 2)),
                dataloader_multiprocessing_context=str(
                    self.vision_config.get("dataloader_multiprocessing_context", "spawn")
                ),
                dataloader_persistent_workers=bool(self.vision_config.get("dataloader_persistent_workers", True)),
                device=str(self.vision_config.get("decoder_device", "cuda")),
                video_override_map=self.vision_config.get("video_override_map"),
            )
            log.info("Framework CUDA TorchCodec video preprocessor initialized")
        else:
            raise ValueError(f"Unsupported evaluation video decoder: {decoder_name}")

        model = CosmosFrameworkRuntime(
            self.model_config.get("model_name"),
            config_file=self.model_config.get("config_file"),
            export_dir=self.model_config.get("export_dir"),
            vit_checkpoint_path=self.model_config.get("vit_checkpoint_path"),
            enable_lora=self.enable_lora or self.model_config.get("enable_lora", False),
            base_model_path=self.model_config.get("base_model_path"),
            dtype=self.model_config.get("dtype", "bfloat16"),
            device_map=self.model_config.get("device_map", "auto"),
            attn_implementation=self.model_config.get("attn_implementation"),
        )
        processor = model.processor

        elapsed_time = time.time() - start_time
        log.info(f"Model loaded in {elapsed_time:.2f} seconds")
        self._send_status_callback(f"Model loaded successfully in {elapsed_time:.1f} seconds")
        return model, processor

    def prepare_inputs_parallel(
        self,
        input_tasks: List[Dict[str, Any]],
        num_processes: int,
    ) -> List[Any]:
        """Prepare model inputs for tasks in parallel."""
        # Framework processors decode videos while constructing each inference
        # batch.  Returning lightweight task dictionaries avoids materialising
        # an entire validation shard's decoded frames in host RAM.
        if self.video_preprocessor is not None:
            # Decode only the current inference batch in the persistent spawned
            # worker; keeping this list lightweight avoids retaining the whole
            # validation shard's decoded frames in the evaluator process.
            return input_tasks
        return input_tasks

    def _prepare_single_model_input(
        self,
        input_task: Dict[str, Any],
        processor: Any,
        vision_config: Dict[str, Any],
    ) -> Optional[Any]:
        """Prepare input data for a single model inference task."""
        return input_task

    def run_model_inference(
        self,
        inputs: List[Any],
        input_tasks: List[Dict[str, Any]],
        answer_type: str = "freeform",
    ) -> Tuple[List[str], List[float]]:
        """Run the model on inputs and return predictions and per-sample losses.

        Returns:
            Tuple containing:
            - List of prediction strings
            - List of per-sample losses (negative log-likelihood)
        """
        generation_config = dict(self.gen_config)
        generation_config.setdefault("seed", self.eval_config.get("seed", 1))
        if answer_type == "letter":
            generation_config["max_tokens"] = min(int(generation_config.get("max_tokens", 10)), 10)
            generation_config["temperature"] = 0.0

        log.info(f"Generating outputs for {len(inputs)} tasks using Cosmos Framework...")
        self._send_status_callback(f"Starting model inference for {len(inputs)} tasks...")

        inference_start = time.time()

        # Process in batches to send status callbacks during long generation
        # This prevents timeout for large datasets where generation can take >15 minutes
        batch_size = self.eval_config.get("batch_size", 8)
        log.info(f"Using batch size: {batch_size}")
        all_outputs = []
        all_losses = []

        total_batches = (len(inputs) + batch_size - 1) // batch_size
        progress_interval_batches = max(1, int(self.eval_config.get("progress_interval_batches", 1)))
        if self.video_preprocessor is not None:
            batches = self.video_preprocessor.iter_prepared_batches(inputs, batch_size=batch_size)
        else:
            batches = (inputs[index : index + batch_size] for index in range(0, len(inputs), batch_size))

        for batch_num, batch_inputs in enumerate(batches, start=1):
            batch_idx = (batch_num - 1) * batch_size
            batch_end = min(batch_idx + len(batch_inputs), len(inputs))
            report_progress = batch_num == 1 or batch_num == total_batches or batch_num % progress_interval_batches == 0

            if report_progress:
                log.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_inputs)} requests)...")
                self._send_status_callback(
                    f"Model inference: Processing batch {batch_num}/{total_batches} "
                    f"(requests {batch_idx + 1}-{batch_end}/{len(inputs)})"
                )

            batch_start_time = time.time()
            batch_predictions, batch_losses = self.model.generate_tasks(
                batch_inputs,
                generation_config=generation_config,
                vision_config=self.vision_config,
            )
            batch_time = time.time() - batch_start_time

            all_outputs.extend(batch_predictions)
            all_losses.extend(batch_losses)

            completed = len(all_outputs)
            progress_pct = (completed / len(inputs)) * 100
            elapsed_total = time.time() - inference_start

            if report_progress:
                log.info(
                    f"Batch {batch_num}/{total_batches} completed in {batch_time:.1f}s. "
                    f"Total progress: {completed}/{len(inputs)} ({progress_pct:.1f}%)"
                )
                self._send_status_callback(
                    f"Model inference progress: {completed}/{len(inputs)} requests completed "
                    f"({progress_pct:.1f}%), elapsed time: {elapsed_total:.1f}s"
                )

        inference_time = time.time() - inference_start
        log.info(f"Finished Framework generation. Received {len(all_outputs)} outputs in {inference_time:.1f}s.")
        cache_stats = getattr(self.model, "video_cache_stats", lambda: {})()
        if cache_stats:
            print(
                "COSMOS_VIDEO_RUNTIME_CACHE_SUMMARY "
                + " ".join(f"{key}={value}" for key, value in sorted(cache_stats.items())),
                flush=True,
            )
        self._send_status_callback(
            f"Model inference completed: {len(all_outputs)} outputs generated in {inference_time:.1f} seconds"
        )

        predictions = all_outputs
        losses = all_losses
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        log.info(f"Average evaluation loss (NLL): {avg_loss:.6f}")
        self._send_status_callback(f"Successfully extracted {len(predictions)} predictions, avg loss: {avg_loss:.4f}")
        return predictions, losses

    def run_evaluation(
        self,
        results_dir: Path,
        skip_saved: bool = False,
        limit: int = -1,
        total_shard: int = 1,
        shard_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.

        Args:
            results_dir: Directory to save results
            skip_saved: Whether to skip already saved results
            limit: Limit number of tasks (for debugging)
            total_shard: Total number of shards
            shard_id: Current shard ID

        Returns:
            Dictionary containing evaluation metrics
        """
        start_time = time.time()

        # Load model
        self._send_status_callback("Initializing evaluation pipeline...")
        self.model, self.processor = self.load_model()

        # Get evaluation parameters
        answer_type = self.eval_config.get("answer_type", "freeform")
        num_processes = self.eval_config.get("num_processes", 40)

        # Create results directory structure
        save_folder = self.model_config.get("save_folder", None)
        if save_folder:
            results_output_dir = results_dir / save_folder
        else:
            model_name = self.model_config.get("model_name", "unknown_model")
            results_output_dir = results_dir / Path(model_name).name / answer_type

        results_output_dir.mkdir(parents=True, exist_ok=True)

        # Under torchrun each process owns exactly one data-parallel shard and one
        # GPU.  A non-distributed invocation evaluates its requested shard.
        shard_ids_to_run = [shard_id]
        all_outputs: List[Dict[str, Any]] = []
        all_predictions: List[str] = []
        all_losses: List[float] = []

        for current_shard_id in shard_ids_to_run:
            # Step 1: Gather tasks for this shard
            log.info(f"Gathering evaluation tasks (shard {current_shard_id + 1}/{total_shard})...")
            self._send_status_callback(
                f"Gathering evaluation tasks from dataset (shard {current_shard_id}/{total_shard})..."
            )
            inputs, outputs = self.make_tasks(results_output_dir, total_shard, current_shard_id)
            log.info(f"Gathered {len(inputs)} tasks for shard {current_shard_id}")

            log_workflow_status(
                data={
                    "evaluation_phase": "task_gathering",
                    "total_tasks": len(inputs),
                    "shard_id": current_shard_id,
                    "total_shards": total_shard,
                    "lora_enabled": self.enable_lora or self.model_config.get("enable_lora", False),
                },
                component_name=COMPONENT_NAME,
            )

            # Step 2: Skip saved results if requested
            if skip_saved:
                filtered_inputs = []
                filtered_outputs = []
                for i, o in zip(inputs, outputs):
                    if not Path(o["output_path"]).exists():
                        filtered_inputs.append(i)
                        filtered_outputs.append(o)
                inputs, outputs = filtered_inputs, filtered_outputs
                if not inputs:
                    log.info(f"Shard {current_shard_id}: all results already saved, skipping")
                    continue

            # Apply limit if specified (per-shard limit would be complex; we apply to first shard only for debugging)
            if limit > 0 and len(inputs) > limit:
                inputs = inputs[:limit]
                outputs = outputs[:limit]
                log.info(f"Limited tasks to {len(inputs)} for debugging")

            if not inputs:
                continue

            # Step 3: Prepare model inputs
            log.info(f"Preparing model inputs for shard {current_shard_id}...")
            prepared = self.prepare_inputs_parallel(inputs, num_processes)
            log_workflow_status(
                data={"evaluation_phase": "input_preparation", "prepared_inputs": len(prepared)},
                component_name=COMPONENT_NAME,
            )

            # Step 4: Run model inference
            log.info(f"Running model inference for shard {current_shard_id}...")
            inference_start = time.time()
            predictions, losses = self.run_model_inference(prepared, inputs, answer_type)
            inference_time = time.time() - inference_start
            log.info(f"Shard {current_shard_id} inference completed in {inference_time:.2f}s")
            log_workflow_status(
                data={
                    "evaluation_phase": "inference",
                    "inference_time_seconds": inference_time,
                    "tasks_processed": len(inputs),
                    "avg_loss": sum(losses) / len(losses) if losses else 0.0,
                },
                component_name=COMPONENT_NAME,
            )

            # Step 5: Save results for this shard
            log.info(f"Saving results for shard {current_shard_id}...")
            self._current_shard_id = current_shard_id
            self._current_total_shard = total_shard
            self.save(outputs, predictions)
            all_outputs.extend(outputs)
            all_predictions.extend(predictions)
            all_losses.extend(losses)

        if not all_outputs:
            log.info("No tasks to evaluate - all results already exist or no data.")
            self._send_status_callback("No tasks to evaluate - all results already exist")
            return {"overall": {"accuracy": 0.0, "total": 0, "correct": 0}}

        # Step 6: Compute overall metrics from all shards
        log.info("Computing evaluation metrics over full dataset...")
        self._send_status_callback("Computing evaluation metrics...")
        metrics = self.compute_metrics(results_output_dir, all_outputs, all_predictions)

        # Step 7: Add loss to metrics (common for all evaluators)
        # This is computed from logprobs during inference and applies to all task types
        if all_losses:
            avg_loss = sum(all_losses) / len(all_losses)
            if "overall" in metrics:
                metrics["overall"]["loss"] = avg_loss
            else:
                metrics["overall"] = {"loss": avg_loss}
            log.info(f"Evaluation loss (NLL): {avg_loss:.6f}")

        # Coordinate independent data-parallel ranks without creating a torch
        # process group.  This keeps Framework checkpoint loading local to each
        # GPU and avoids DCP treating evaluation replicas as one sharded model.
        rank, world_size, _ = distributed_identity()
        if world_size > 1:
            marker_dir = results_output_dir / ".rank_status"
            marker_dir.mkdir(parents=True, exist_ok=True)
            (marker_dir / f"rank_{rank}.done").write_text("done\n")
            if rank == 0:
                timeout = int(self.eval_config.get("barrier_timeout_seconds", 14400))
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if all((marker_dir / f"rank_{i}.done").is_file() for i in range(world_size)):
                        break
                    time.sleep(1)
                else:
                    missing = [i for i in range(world_size) if not (marker_dir / f"rank_{i}.done").is_file()]
                    raise TimeoutError(f"Timed out waiting for evaluation ranks: {missing}")
                aggregated = self.aggregate_metrics_if_all_shards_present(results_output_dir, metrics, world_size, rank)
                if aggregated is not None:
                    metrics = aggregated
            # Nonzero ranks return their local metric; only rank zero publishes
            # the aggregate through the action/status logger.

        if rank == 0:
            import json

            (results_output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

        total_time = time.time() - start_time
        log.info(f"Complete evaluation pipeline finished in {total_time:.2f} seconds")
        self._send_status_callback(f"Evaluation completed successfully in {total_time:.1f} seconds")

        return metrics

    @abstractmethod
    def make_tasks(
        self, results_dir: Path, total_shard: int, shard_id: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Create evaluation tasks from the dataset."""
        ...

    @abstractmethod
    def save(self, outputs: List[Dict[str, Any]], predictions: List[str]) -> None:
        """Save predictions to output files."""
        ...

    @abstractmethod
    def compute_metrics(
        self, results_dir: Path, outputs: List[Dict[str, Any]], predictions: List[str]
    ) -> Dict[str, Any]:
        """Compute evaluation metrics from predictions."""
        ...

    def aggregate_metrics_if_all_shards_present(
        self,
        results_dir: Path,
        current_metrics: Dict[str, Any],
        total_shard: int,
        shard_id: int,
    ) -> Optional[Dict[str, Any]]:
        """If all shards have written results, load and aggregate to overall metrics. Default: no-op."""
        return None
