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
"""General evaluator for text-generation tasks (captioning, freeform QA)."""

from __future__ import annotations

import json
import logging as log
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cosmos_framework.evaluation.reasoner.base import BaseEvaluator
from cosmos_framework.evaluation.reasoner.metrics.binary_classification import (
    compute_binary_classification_metrics,
    compute_saved_binary_metrics,
    extract_binary_response,
)
from cosmos_framework.evaluation.reasoner.metrics.task_accuracy import (
    score_accuracy,
    score_saved_results,
)
from cosmos_framework.evaluation.reasoner.metrics.text_metrics import TextMetrics
from cosmos_framework.evaluation.reasoner.sharding import media_balanced_shard

COMPONENT_NAME = "Cosmos Framework Evaluation"

POSITIVE_LABELS = {"yes", "a"}
NEGATIVE_LABELS = {"no", "b"}
BINARY_LABELS = POSITIVE_LABELS | NEGATIVE_LABELS
MCQ_LABELS = {"a", "b", "c", "d"}


class Evaluator(BaseEvaluator):
    """
    General evaluator for text-generation tasks (captioning, freeform QA).
    - Uses the common BaseEvaluator pipeline
    - Computes text metrics via Hugging Face 'evaluate' (BLEU/ROUGE)
    """

    def __init__(self, config: Dict[str, Any], enable_lora: bool = False) -> None:
        """
        Initialize the Evaluator.
        """
        super().__init__(config, enable_lora=enable_lora)
        metrics_cfg = config.get("metrics", {})

        # Get metric names as list
        metric_names = metrics_cfg.get("names", ["bleu", "rouge"])
        # Handle legacy comma-separated strings for backward compatibility
        if isinstance(metric_names, str):
            metric_names = [name.strip() for name in metric_names.split(",")]

        self.metrics = TextMetrics(
            metrics=metric_names,
            bertscore_model=metrics_cfg.get("bertscore_model", "microsoft/deberta-xlarge-mnli"),
            bertscore_lang=metrics_cfg.get("bertscore_lang", "en"),
            bertscore_device=metrics_cfg.get("bertscore_device"),
            bertscore_batch_size=metrics_cfg.get("bertscore_batch_size"),
        )

    def make_tasks(
        self, results_dir: Path, total_shard: int, shard_id: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Make tasks for the Evaluator.
        """
        annotation_path = self.dataset_cfg.get("annotation_path")
        media_dir = self.dataset_cfg.get("media_dir", None)
        system_prompt = self.dataset_cfg.get("system_prompt", "")

        with open(annotation_path, "r") as f:
            annotation_payload = json.load(f)
        if isinstance(annotation_payload, list):
            annotations = annotation_payload
            annotation_metadata = {}
        elif isinstance(annotation_payload, dict) and isinstance(annotation_payload.get("items"), list):
            annotations = annotation_payload["items"]
            annotation_metadata = annotation_payload.get("metadata", {}) or {}
            embedded_media_root = annotation_payload.get("media_root")
            if media_dir is None and embedded_media_root:
                media_dir = embedded_media_root
        else:
            raise ValueError("annotation_path must contain a JSON array or a task-aware object with an items array")

        configured_task = str(self.config.get("task", {}).get("type", "")).lower()

        def task_for(record: Dict[str, Any]) -> str:
            task = record.get("task") or record.get("task_type") or annotation_metadata.get("task") or configured_task
            return str(task).strip().lower().replace("-", "_").replace(" ", "_")

        def resolve_media_paths(record: Dict[str, Any]) -> Tuple[List[str], str]:
            images = record.get("image", None) or record.get("images", None)
            videos = record.get("video", None) or record.get("video_id", None)
            if images:
                if isinstance(images, str):
                    images = [images]
                rel = images
                mode = "image"
            elif videos:
                if isinstance(videos, str):
                    videos = [videos]
                rel = videos
                mode = "video"
            else:
                rel = []
                mode = "image"
            if media_dir:
                paths = [os.path.join(media_dir, p) for p in rel]
            else:
                paths = rel
            return paths, mode

        qa_pairs: List[Dict[str, Any]] = []
        for item in annotations:
            if "conversations" in item:
                question = re.sub(r"(\n)?</?(image|video)>(\n)?", "", item["conversations"][0]["value"]).strip()
                answer = item["conversations"][1]["value"]
                refs = [answer]
                media_paths, media_mode = resolve_media_paths(item)
                qa_pairs.append(
                    {
                        "id": item["id"],
                        "question": question,
                        "prompt": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ],
                        "references": refs,
                        "task": task_for(item),
                        "media_paths": media_paths,
                        "media_mode": media_mode,
                    }
                )
            else:
                rid = item.get("id", item.get("media_id", item.get("video_id", "")))
                prompt = item.get("prompt", item.get("question", ""))
                refs = item.get("references", item.get("answer", []))
                media_paths, media_mode = resolve_media_paths(item)
                qa_pairs.append(
                    {
                        "id": rid,
                        "question": prompt,
                        "prompt": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "references": refs if isinstance(refs, list) else [str(refs)],
                        "task": task_for(item),
                        "media_paths": media_paths,
                        "media_mode": media_mode,
                    }
                )

        shard_strategy = str(self.eval_config.get("shard_strategy", "stride"))
        if shard_strategy == "media_balanced":
            shard = media_balanced_shard(qa_pairs, max(1, total_shard), shard_id)
        elif shard_strategy == "stride":
            shard = qa_pairs[shard_id :: max(1, total_shard)]
        else:
            raise ValueError(f"Unsupported evaluation shard_strategy: {shard_strategy}")
        tasks: List[Dict[str, Any]] = []
        outs: List[Dict[str, Any]] = []
        for rec in shard:
            out_path = results_dir / "general" / f"{rec['id']}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append(rec)
            outs.append(
                {
                    "id": rec["id"],
                    "question": rec["question"],
                    "output_path": str(out_path),
                    "references": rec["references"],
                    "task": rec["task"],
                }
            )
        return tasks, outs

    def save(self, outputs: List[Dict[str, Any]], predictions: List[str]) -> None:
        """
        Save all outputs and predictions to a single aggregated JSON file.

        For binary tasks (yes/no, a/b), the format includes a short extracted
        label as ``response`` and the raw model output as ``full_response``.
        For other tasks, ``response`` contains the raw model output.
        """
        references_by_output = [o["references"] for o in outputs]
        is_binary = all(
            any(ref.strip().lower() in BINARY_LABELS for ref in (r if isinstance(r, list) else [r]))
            for r in references_by_output
            if r
        )
        is_mcq = (
            not is_binary
            and bool(references_by_output)
            and all(
                all(ref.strip().lower() in MCQ_LABELS for ref in (refs if isinstance(refs, list) else [refs]))
                for refs in references_by_output
                if refs
            )
        )

        all_results = []
        for out, pred in zip(outputs, predictions):
            gt = out["references"]
            if isinstance(gt, list) and len(gt) == 1:
                gt = gt[0]

            if is_binary:
                all_results.append(
                    {
                        "video_id": out["id"],
                        "question": out["question"],
                        "response": self._extract_binary_response(pred),
                        "gt": gt.lower() if isinstance(gt, str) else gt,
                        "full_response": pred,
                        "task": out.get("task", "binary"),
                    }
                )
            elif is_mcq:
                all_results.append(
                    {
                        "video_id": out["id"],
                        "question": out["question"],
                        "response": self._extract_mcq_response(pred),
                        "gt": gt.upper() if isinstance(gt, str) else gt,
                        "full_response": pred,
                        "task": out.get("task", "mcq"),
                    }
                )
            else:
                all_results.append(
                    {
                        "video_id": out["id"],
                        "question": out["question"],
                        "response": pred,
                        "gt": gt,
                        "task": out.get("task", ""),
                    }
                )

        results_dir = Path(outputs[0]["output_path"]).parent
        results_dir.mkdir(parents=True, exist_ok=True)
        total_shard = getattr(self, "_current_total_shard", 1)
        shard_id = getattr(self, "_current_shard_id", 0)
        if total_shard > 1:
            out_path = results_dir / f"results_shard{shard_id}.json"
        else:
            out_path = results_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"Saved {len(all_results)} results to {out_path}")

    @staticmethod
    def _normalize_text(text: str, is_mcq: bool = False) -> str:
        """
        Normalize text for exact-match comparison:
        - Lowercase
        - Remove punctuation and symbols (Unicode-aware)
        - Remove articles (a, an, the) - only for non-MCQ tasks
        - Collapse multiple spaces

        Args:
            text: The text to normalize
            is_mcq: If True, skip article removal to preserve single-letter answers (A, B, C, D)
        """
        if not isinstance(text, str):
            text = str(text)
        s = text.lower()
        # Remove punctuation and symbols using Unicode categories (P*, S*)
        s = "".join(ch if not unicodedata.category(ch).startswith(("P", "S")) else " " for ch in s)
        # Remove English articles only for non-MCQ tasks
        # For MCQ, we need to preserve single letters like "a", "b", "c", "d"
        if not is_mcq:
            s = re.sub(r"\b(a|an|the)\b", " ", s)
        # Collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def _is_exact_match(cls, prediction: str, references: List[str], is_mcq: bool = False) -> bool:
        """
        Check if the prediction is an exact match for any of the references.

        Args:
            prediction: The model's prediction
            references: List of reference answers
            is_mcq: If True, use MCQ-friendly normalization (preserves single letters)
        """
        pred_norm = cls._normalize_text(prediction, is_mcq=is_mcq)
        for ref in references:
            if cls._normalize_text(ref, is_mcq=is_mcq) == pred_norm:
                return True
        return False

    @staticmethod
    def _extract_binary_response(prediction: str) -> Optional[str]:
        """Extract one unambiguous canonical yes/no label from free-form text."""
        return extract_binary_response(prediction)

    @staticmethod
    def _extract_mcq_response(prediction: str) -> str:
        """Extract a single A-D choice from an MCQ prediction."""
        text = prediction
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        match = re.search(r"\b([A-D])\b", text.strip(), re.IGNORECASE)
        return match.group(1).upper() if match else text.strip().upper()

    @staticmethod
    def _compute_binary_classification_metrics(
        predictions: List[str],
        references: List[List[str]],
    ) -> Dict[str, Any]:
        """Compute binary classification metrics (TP/FP/TN/FN and derived scores).

        Positive labels: "yes", "a"
        Negative labels: "no", "b"
        """
        return compute_binary_classification_metrics(predictions, references)

    def compute_metrics(
        self, results_dir: Path, outputs: List[Dict[str, Any]], predictions: List[str]
    ) -> Dict[str, Any]:
        """
        Compute the metrics for the Evaluator.
        """
        self._send_status_callback("Computing text similarity metrics (BLEU/ROUGE/BERTScore)...")
        references: List[List[str]] = [o["references"] for o in outputs]

        # Compute similarity metrics (BLEU/ROUGE/BERTScore)
        metrics = self.metrics.compute(predictions=predictions, references=references)
        self._send_status_callback("Text similarity metrics computed")

        # Detect task type from config, task-aware metadata, or references.
        task_type = self.config.get("task", {}).get("type", "")
        output_tasks = [str(output.get("task") or task_type).lower().replace("-", "_") for output in outputs]

        # Detect binary task: references are all yes/no or a/b
        is_binary = task_type.lower() == "binary"
        if not is_binary and references:
            is_binary = all(any(ref.strip().lower() in BINARY_LABELS for ref in refs) for refs in references if refs)

        # Detect MCQ task: references are single A-D letters.
        is_mcq = task_type.lower() == "mcq"
        if not is_mcq and not is_binary and references:
            is_mcq = all(all(ref.strip().lower() in MCQ_LABELS for ref in refs) for refs in references if refs)

        scored_predictions = [self._extract_mcq_response(pred) if is_mcq else pred for pred in predictions]

        normalized_tasks = [
            task or ("bcq" if is_binary else "mcq" if is_mcq else "unspecified") for task in output_tasks
        ]
        accuracy_result = score_accuracy(predictions, references, normalized_tasks)
        correct = accuracy_result["overall"]["correct"]
        total = accuracy_result["overall"]["total"]
        accuracy = accuracy_result["overall"]["accuracy"]

        # Optional soft accuracy via token-overlap F1 threshold
        def _tokens(s: str) -> List[str]:
            return self._normalize_text(s, is_mcq=is_mcq).split()

        def _f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
            if not pred_tokens and not ref_tokens:
                return 1.0
            if not pred_tokens or not ref_tokens:
                return 0.0
            common = {}
            for t in pred_tokens:
                common[t] = common.get(t, 0) + 1
            overlap = 0
            for t in ref_tokens:
                if common.get(t, 0) > 0:
                    overlap += 1
                    common[t] -= 1
            if overlap == 0:
                return 0.0
            precision = overlap / len(pred_tokens)
            recall = overlap / len(ref_tokens)
            return 2 * precision * recall / (precision + recall)

        soft_cfg = self.eval_config.get("soft_accuracy", {}) if isinstance(self.eval_config, dict) else {}
        soft_enabled = bool(soft_cfg.get("enabled", True))
        soft_threshold = float(soft_cfg.get("f1_threshold", 0.8))

        soft_correct = 0
        if soft_enabled:
            for pred, refs in zip(scored_predictions, references):
                ptoks = _tokens(pred)
                max_f1 = 0.0
                for ref in refs:
                    rtoks = _tokens(ref)
                    max_f1 = max(max_f1, _f1(ptoks, rtoks))
                if max_f1 >= soft_threshold:
                    soft_correct += 1
        soft_accuracy = float(soft_correct) / max(1, len(predictions)) if soft_enabled else 0.0

        result: Dict[str, Any] = {
            "overall": {
                "accuracy": accuracy,
                "total": total,
                "correct": correct,
                "soft_accuracy": soft_accuracy if soft_enabled else 0.0,
                "soft_correct": soft_correct if soft_enabled else 0,
                "soft_threshold": soft_threshold if soft_enabled else None,
            }
        }
        result.update({key: value for key, value in accuracy_result.items() if key != "overall"})
        result.update(metrics)

        if is_binary and total == len(predictions):
            self._send_status_callback("Computing binary classification metrics...")
            binary_metrics = self._compute_binary_classification_metrics(predictions, references)
            result["binary_metrics"] = binary_metrics
            binary_correct = binary_metrics["TP"] + binary_metrics["TN"]
            result["overall"]["accuracy"] = binary_metrics["accuracy"] / 100.0
            result["overall"]["correct"] = binary_correct
            result["overall"]["balanced_accuracy"] = binary_metrics["balanced_accuracy"]
            result["overall"]["f1_score"] = binary_metrics["f1_score"]
            log.info(
                f"Binary metrics - Accuracy: {binary_metrics['accuracy']}%, "
                f"Balanced Acc: {binary_metrics['balanced_accuracy']}%, "
                f"F1: {binary_metrics['f1_score']}%, "
                f"TP: {binary_metrics['TP']}, FP: {binary_metrics['FP']}, "
                f"TN: {binary_metrics['TN']}, FN: {binary_metrics['FN']}"
            )

        return result

    def aggregate_metrics_if_all_shards_present(
        self,
        results_dir: Path,
        current_metrics: Dict[str, Any],
        total_shard: int,
        shard_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Load all results_shard*.json; if all shards present, merge and return overall metrics."""
        import glob

        # General evaluator writes under results_dir / "general" / results_shard*.json
        subdir = results_dir / "general"
        pattern = str(subdir / "**" / "results_shard*.json")
        files = sorted(glob.glob(pattern, recursive=True))
        if len(files) != total_shard:
            log.debug(f"Aggregate: found {len(files)} shard files, need {total_shard}; skipping overall")
            return None
        all_results = []
        for path in files:
            with open(path, "r") as f:
                all_results.extend(json.load(f))
        if not all_results:
            return None
        aggregated = score_saved_results(all_results)
        total = aggregated["overall"]["total"]
        correct = aggregated["overall"]["correct"]

        # Text metrics are not rank-reducible. Preserve only scalar metrics that
        # are unrelated to task accuracy; never overwrite global task metadata
        # with a rank-local value.
        task_metric_keys = {
            "overall",
            "binary_metrics",
            "per_task",
            "aggregation",
            "coverage",
            "excluded_tasks",
            "evaluator_version",
        }
        for key, value in current_metrics.items():
            if key not in task_metric_keys:
                aggregated[key] = value
        # Binary: recompute from merged response/gt
        covered_results = [
            result
            for result in all_results
            if str(result.get("task", "")).lower().replace("-", "_") in {"bcq", "binary"}
        ]
        is_binary = bool(covered_results) and len(covered_results) == total
        if is_binary and covered_results:
            binary_metrics = compute_saved_binary_metrics(covered_results)
            aggregated["binary_metrics"] = binary_metrics
            aggregated["overall"]["accuracy"] = binary_metrics["accuracy"] / 100.0
            aggregated["overall"]["correct"] = binary_metrics["TP"] + binary_metrics["TN"]
            aggregated["overall"]["balanced_accuracy"] = binary_metrics["balanced_accuracy"]
            aggregated["overall"]["f1_score"] = binary_metrics["f1_score"]
        log.info(
            f"Aggregated overall metrics from {total_shard} shards: "
            f"accuracy={aggregated['overall']['accuracy']}, total={total}, correct={correct}"
        )
        return aggregated
