# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Internal GEPA batch entry point; reuse native model, decoding and output handling.

Reflection minibatches may contain just one class. The normal evaluator's
balanced-accuracy diagnostic requires both classes, so only metric aggregation
is specialized here. This module does not train or implement another inference engine.
"""

import argparse
from pathlib import Path

from cosmos_framework.automl.evaluate_trial import read_toml
from cosmos_framework.automl.prompt_metrics import aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from cosmos_framework.evaluation.reasoner.evaluator import Evaluator

    class PromptEvaluator(Evaluator):
        def compute_metrics(self, results_dir, outputs, predictions):
            golds = [{"references": row["references"], "task": row["task"]} for row in outputs]
            return {"overall": aggregate(predictions, golds), "evaluator_version": "cosmos-prompts-v1"}

    config = read_toml(args.config)
    evaluator = PromptEvaluator(config, enable_lora=config.get("model", {}).get("enable_lora", False))
    evaluator.run_evaluation(Path(config["results_dir"]), skip_saved=False, total_shard=1, shard_id=0)


if __name__ == "__main__":
    main()
