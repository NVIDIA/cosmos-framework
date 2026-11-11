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

"""ITS Evaluation script for Cosmos Framework.

Example:

```shell
cosmos-reasoner-evaluate --config evaluation.toml
```
"""

import argparse
import logging
from pathlib import Path

import toml

from cosmos_framework.checkpoint.reasoner import distributed_identity
from cosmos_framework.evaluation.reasoner.status_barrier import wait_for_status_publishers
from cosmos_framework.utils.workflow_status import (
    Status,
    Verbosity,
    get_status_logger,
    log_workflow_status,
    monitor_status,
)

# Constants
COMPONENT_NAME = "Cosmos Framework Evaluation"
SEPARATOR = "-" * 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cosmos Framework Evaluation Script")

    parser.add_argument("--config", "-c", type=str, required=True, help="Path to evaluation configuration TOML file")
    return parser.parse_args()


def _fmt_accuracy(value):
    """Accuracy is None for generative-only runs with no accuracy-defined tasks."""
    return "n/a" if value is None else f"{value:.4f}"


def run_evaluation(args):
    """
    Run the evaluation pipeline.

    Args:
        args: Parsed command line arguments
    """

    # Get status logger for Cosmos integration
    s_logger = get_status_logger()

    try:
        s_logger.write(status_level=Status.RUNNING, message="Starting evaluation...", verbosity_level=Verbosity.INFO)

        # Load configuration
        config_path = Path(args.config)
        logger.info(f"Loading configuration from {config_path}")

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            eval_config = toml.load(f)

        # Framework evaluation is data parallel: launch with torchrun and each
        # rank owns one GPU/model replica and one validation shard.
        rank, world_size, local_rank = distributed_identity()
        if "evaluation" not in eval_config:
            eval_config["evaluation"] = {}
        eval_config["evaluation"]["total_shard"] = world_size
        eval_config["evaluation"]["shard_id"] = rank
        logger.info(f"Using Framework data parallelism: rank={rank}, local_rank={local_rank}, world_size={world_size}")

        # Create results directory
        results_dir = eval_config.get("results_dir", "/results")
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to: {results_dir}")

        # Initialize evaluator
        task_type = eval_config.get("task", {}).get("type", "its_directionality")
        logger.info(f"Initializing evaluator for task: {task_type}")
        lora_enabled = eval_config.get("model", {}).get("enable_lora", False)
        if task_type == "its_directionality":
            from cosmos_framework.evaluation.reasoner.its_evaluator import ITSEvaluator

            evaluator = ITSEvaluator(eval_config, enable_lora=lora_enabled)
        elif task_type == "metropolis_sgd":
            from cosmos_framework.evaluation.reasoner.metropolis_sgd_evaluator import MetropolisSGDEvaluator

            evaluator = MetropolisSGDEvaluator(eval_config, enable_lora=lora_enabled)
        else:
            from cosmos_framework.evaluation.reasoner.evaluator import Evaluator

            evaluator = Evaluator(eval_config, enable_lora=lora_enabled)

        # Log evaluation start to Cosmos
        log_workflow_status(
            data={
                "evaluation_status": "started",
                "config": str(config_path),
                "results_dir": str(results_dir),
                "model_name": eval_config.get("model", {}).get("model_name", "unknown"),
            },
            component_name=COMPONENT_NAME,
        )

        logger.info(SEPARATOR)
        logger.info("STARTING EVALUATION PIPELINE")
        logger.info(SEPARATOR)
        logger.info(f"Task: {task_type}")
        logger.info(f"Config: {config_path}")
        logger.info(f"Results: {results_dir}")
        logger.info(f"Model: {eval_config.get('model', {}).get('model_name', 'unknown')}")
        logger.info(SEPARATOR)

        # Get evaluation parameters from config
        eval_params = eval_config.get("evaluation", {})
        skip_saved = eval_params.get("skip_saved", False)
        limit = eval_params.get("limit", -1)
        total_shard = eval_params.get("total_shard", 1)
        shard_id = eval_params.get("shard_id", 0)

        # Run evaluation pipeline (inference + scoring)
        logger.info("Starting evaluation pipeline...")
        results = evaluator.run_evaluation(
            results_dir=results_dir, skip_saved=skip_saved, limit=limit, total_shard=total_shard, shard_id=shard_id
        )

        # Extract metrics
        overall_metrics = results.get("overall", {})
        overall_accuracy = overall_metrics.get("accuracy", 0.0)
        total_samples = overall_metrics.get("total", 0)
        correct_samples = overall_metrics.get("correct", 0)

        # Prepare KPI data with all metrics flattened for visibility
        kpi_data = {
            "evaluation_status": "completed",
            "accuracy": overall_accuracy,
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "results_path": str(results_dir),
        }

        # Add evaluation loss if present
        if "loss" in overall_metrics:
            kpi_data["loss"] = overall_metrics["loss"]

        # Add soft accuracy metrics if present
        if "soft_accuracy" in overall_metrics:
            kpi_data["soft_accuracy"] = overall_metrics["soft_accuracy"]
        if "soft_correct" in overall_metrics:
            kpi_data["soft_correct"] = overall_metrics["soft_correct"]
        if "soft_threshold" in overall_metrics:
            kpi_data["soft_threshold"] = overall_metrics["soft_threshold"]

        # Add weighted score if present (Metropolis SGD)
        if "weighted_score" in overall_metrics:
            kpi_data["weighted_score"] = overall_metrics["weighted_score"]

        # Add binary classification metrics if present
        if "balanced_accuracy" in overall_metrics:
            kpi_data["balanced_accuracy"] = overall_metrics["balanced_accuracy"]
        if "f1_score" in overall_metrics:
            kpi_data["f1_score"] = overall_metrics["f1_score"]
        binary_metrics = results.get("binary_metrics")
        if binary_metrics:
            for bkey in (
                "TP",
                "FP",
                "TN",
                "FN",
                "precision",
                "recall",
                "positive_accuracy",
                "negative_accuracy",
                "total_samples",
                "parsed_predictions",
                "unparseable_predictions",
                "unparseable_positive_predictions",
                "unparseable_negative_predictions",
                "positive_ground_truth",
                "negative_ground_truth",
            ):
                kpi_data[bkey] = binary_metrics[bkey]

        # Add text similarity metrics (BLEU, ROUGE, BERTScore) to top level
        for key, value in results.items():
            if key not in ["overall", "metrics"] and not isinstance(value, dict):
                # Add top-level metrics like BLEU, ROUGE1, ROUGE2, etc.
                kpi_data[key.lower()] = value

        # Add per-category metrics (for Metropolis SGD, ITS, etc.)
        for category, cat_metrics in results.items():
            if category not in ["overall", "metrics", "detailed_results"] and isinstance(cat_metrics, dict):
                # Flatten category metrics: e.g., count_accuracy, distance_accuracy
                if "accuracy" in cat_metrics:
                    kpi_data[f"{category}_accuracy"] = cat_metrics["accuracy"]
                if "total" in cat_metrics:
                    kpi_data[f"{category}_total"] = cat_metrics["total"]
                if "correct" in cat_metrics:
                    kpi_data[f"{category}_correct"] = cat_metrics["correct"]

        # Keep detailed_results for backward compatibility
        kpi_data["detailed_results"] = results

        # Log final results to Cosmos
        if rank == 0:
            log_workflow_status(data=kpi_data, component_name=COMPONENT_NAME)

        # Print summary
        if rank == 0:
            print("\n" + "=" * 60)
            print("EVALUATION COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"Results saved to: {results_dir}")
            print(f"Overall accuracy: {_fmt_accuracy(overall_accuracy)}")
            print(f"Total samples: {total_samples}")
            print(f"Correct samples: {correct_samples}")
            if "loss" in overall_metrics:
                print(f"Generation NLL: {overall_metrics['loss']:.6f}")
            if "balanced_accuracy" in overall_metrics:
                print(f"Balanced accuracy: {overall_metrics['balanced_accuracy']:.2f}%")
            if "f1_score" in overall_metrics:
                print(f"F1 score: {overall_metrics['f1_score']:.2f}%")
            reported_metric_names = (
                "BLEU",
                "ROUGE1",
                "ROUGE2",
                "ROUGEL",
                "ROUGELSUM",
                "BERTScore_P",
                "BERTScore_R",
                "BERTScore_F1",
                "WEIGHTED_SCORE",
            )
            for metric_name in reported_metric_names:
                if metric_name in results:
                    print(f"{metric_name}: {results[metric_name]:.6f}")
            for metric_name, metric_value in results.items():
                if (
                    metric_name not in ("overall", "metrics")
                    and metric_name not in reported_metric_names
                    and isinstance(metric_value, (int, float))
                ):
                    print(f"{metric_name}: {metric_value:.6f}")
            print()

        # Print binary classification breakdown if present
        binary_metrics = results.get("binary_metrics")
        if rank == 0 and binary_metrics:
            print("Binary classification metrics:")
            print(f"  TP: {binary_metrics['TP']:<6}  FP: {binary_metrics['FP']}")
            print(f"  TN: {binary_metrics['TN']:<6}  FN: {binary_metrics['FN']}")
            print(f"  Precision:         {binary_metrics['precision']:.2f}%")
            print(f"  Recall:            {binary_metrics['recall']:.2f}%")
            print(f"  Positive Accuracy: {binary_metrics['positive_accuracy']:.2f}%")
            print(f"  Negative Accuracy: {binary_metrics['negative_accuracy']:.2f}%")
            print(
                "  Unparseable:       "
                f"{binary_metrics['unparseable_predictions']} "
                f"(positive={binary_metrics['unparseable_positive_predictions']}, "
                f"negative={binary_metrics['unparseable_negative_predictions']})"
            )
            print()

        # Print per-category results
        if rank == 0:
            print("Per-category results:")
            for category, metrics in results.items():
                if category not in ("overall", "binary_metrics") and isinstance(metrics, dict):
                    accuracy = metrics.get("accuracy", 0.0)
                    total = metrics.get("total", 0)
                    correct = metrics.get("correct", 0)
                    print(f"  {category:<15}: {_fmt_accuracy(accuracy)} ({correct:>3}/{total:<3})")

            print("=" * 60)

        # Every rank emits progress callbacks while evaluating.  Wait until
        # those rank-local publishers are finished, then let rank zero append
        # the one terminal record so a successful job cannot end at RUNNING.
        wait_for_status_publishers(
            results_dir,
            rank,
            world_size,
            int(eval_params.get("barrier_timeout_seconds", 14400)),
        )
        if rank == 0:
            s_logger.write(
                status_level=Status.SUCCESS,
                message=f"Evaluation completed successfully. Accuracy: {_fmt_accuracy(overall_accuracy)}",
                verbosity_level=Verbosity.INFO,
            )

    except KeyboardInterrupt:
        s_logger.write(
            status_level=Status.FAILURE,
            message="Evaluation was interrupted by user (Ctrl+C)",
            verbosity_level=Verbosity.WARNING,
        )
        log_workflow_status(
            data={"evaluation_status": "interrupted", "error": "User interrupted"}, component_name=COMPONENT_NAME
        )
        raise

    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        s_logger.write(status_level=Status.FAILURE, message=error_msg, verbosity_level=Verbosity.ERROR)
        log_workflow_status(data={"evaluation_status": "failed", "error": str(e)}, component_name=COMPONENT_NAME)
        logger.error(error_msg)
        raise


def main():
    """Main entry point for the cosmos-reasoner-evaluate command."""
    args = parse_args()

    # Parse first so --help and argument errors never create run artifacts.
    monitored = monitor_status(name="Cosmos Framework Evaluation", mode="evaluate")(run_evaluation)
    monitored(args)


if __name__ == "__main__":
    main()
