# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller boundary regressions; temporary artifacts, no GPU or network."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import launchable


def prepared(indices=(13404, 13405, 13406, 13407, 13408), media=True):
    return {"dataset_revision": "pinned-test-revision", "manifest_sha256": "scope-and-cuts",
            "annotation_manifest_sha256": "trusted-annotation-and-cuts",
            "media_sha256": {"video_8241": "video-hash"} if media else {},
            "media_manifest_sha256": "media-hash" if media else None,
            "selected_indices": list(indices), "coverage": {
                "expected_rows": 19140, "selected_rows": len(indices),
                "available_rows": len(indices) if media else 0,
                "expected_videos": 5260, "selected_videos": 1, "subset": True}}


def table(work, indices, values=None, stamp="T20261030"):
    output = work / "cosmos" / stamp
    output.mkdir(parents=True)
    path = output / "cosmos_PerceptionTest.tsv"
    values = values if values is not None else ["A"] * len(indices)
    path.write_text("index\tprediction\n" + "".join(
        f"{index}\t{value}\n" for index, value in zip(indices, values)))
    return path


def inference_state(root, selection=None, **updates):
    state = {**(selection or prepared()), "source": launchable.SOURCE,
             "benchmark": "PerceptionTest", "status": "inferred", "scope": "smoke",
             "nframe": 16, "predictions": "native.tsv", "dry_run": False}
    state.update(updates)
    (root / "run.json").write_text(json.dumps(state))
    return state


def scorer_module(effect=None):
    module = types.ModuleType("vlmeval.perception_test_local")
    module.score_predictions = Mock(side_effect=effect)
    return module


class ArtifactTests(unittest.TestCase):
    def test_existing_output_rejected_before_network_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(launchable, "prepare") as prep, \
                patch.object(launchable, "endpoint") as endpoint:
            marker = Path(temp) / "existing"
            marker.write_text("keep")
            with self.assertRaisesRegex(ValueError, "fresh --work-dir"):
                launchable.run(Path(temp), "infer", True, False, Path(temp) / "data")
            prep.assert_not_called()
            endpoint.assert_not_called()
            self.assertEqual(marker.read_text(), "keep")

    def test_invalid_workers_and_frame_protocol_rejected_before_output(self):
        for workers, frames in ((0, 16), (-1, 16), (1, 0), (1, 8), (1, 32)):
            with self.subTest(workers=workers, frames=frames), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "fresh"
                with patch.object(launchable, "prepare") as prep:
                    with self.assertRaises(ValueError):
                        launchable.run(output, "infer", False, False, Path(temp) / "missing",
                                       api_nproc=workers, nframe=frames)
                    prep.assert_not_called()
                self.assertFalse(output.exists())

    def test_coverage_failure_precedes_endpoint_and_output(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / "fresh"
            with patch.object(launchable, "prepare", side_effect=ValueError("missing video_8241")), \
                    patch.object(launchable, "endpoint") as endpoint:
                with self.assertRaisesRegex(ValueError, "missing video_8241"):
                    launchable.run(work, "all", True, False, Path(temp) / "data")
                endpoint.assert_not_called()
            self.assertFalse(work.exists())

    def test_predictions_reject_empty_and_transport_failure_markers(self):
        for value in ("   ", "VIDEO_LOAD_FAILED", "prefix VIDEO_LOAD_FAILED",
                      "Failed to obtain answer via API."):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp:
                path = table(Path(temp), [13404], [value])
                with self.assertRaisesRegex(RuntimeError, "failed/empty"):
                    launchable.verify_predictions(path, [13404])

    def test_nonempty_invalid_and_literal_nan_reach_local_scorer(self):
        for value in ("nan", "ambiguous A or B", "unparseable text"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp:
                path = table(Path(temp), [13404], [value])
                self.assertEqual(launchable.verify_predictions(path, [13404]), path)

    def test_json_null_is_missing_but_literal_nan_is_not(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.json"
            path.write_text(json.dumps([{"index": 13404, "prediction": None}]))
            with self.assertRaisesRegex(RuntimeError, "failed/empty"):
                launchable.verify_predictions(path, [13404])
            path.write_text(json.dumps([{"index": 13404, "prediction": "nan"}]))
            self.assertEqual(launchable.verify_predictions(path, [13404]), path)

    def test_selected_identities_exact_even_when_row_count_matches(self):
        for indices in ([13404, 13404], [13404, 9999], [13404], [13404, 13405, 13406]):
            with self.subTest(indices=indices), tempfile.TemporaryDirectory() as temp:
                path = table(Path(temp), indices)
                with self.assertRaisesRegex(RuntimeError, "identities|expected 2 predictions"):
                    launchable.verify_predictions(path, [13404, 13405])

    def test_multiple_dated_prediction_tables_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            table(work, [13404], stamp="T20261029")
            table(work, [13404], stamp="T20261030")
            with self.assertRaisesRegex(RuntimeError, "expected one prediction table, found 2"):
                launchable.verify_predictions(work, [13404])

    def test_inference_execution_failure_retains_state_config_and_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            work = root / "fresh"
            def fail(command, log_path=None):
                Path(log_path).write_text("diagnostic retained\n")
                raise subprocess.CalledProcessError(3, command)
            with patch.object(launchable, "prepare", return_value=prepared()), \
                    patch.object(launchable, "endpoint"), \
                    patch.object(launchable, "run_command", side_effect=fail):
                with self.assertRaises(subprocess.CalledProcessError):
                    launchable.run(work, "all", True, False, root / "data")
            state = json.loads((work / "run.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["error"]["type"], "CalledProcessError")
            self.assertIn("ended_utc", state["inference_timing"])
            self.assertGreaterEqual(state["inference_timing"]["elapsed_seconds"], 0)
            self.assertEqual((work / "run.log").read_text(), "diagnostic retained\n")
            self.assertTrue((work / "config.json").is_file())

    def test_endpoint_failure_is_retained_and_never_executes_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / "fresh"
            with patch.object(launchable, "prepare", return_value=prepared()), \
                    patch.object(launchable, "endpoint", side_effect=RuntimeError("endpoint unavailable")), \
                    patch.object(launchable, "run_command") as child:
                with self.assertRaisesRegex(RuntimeError, "endpoint unavailable"):
                    launchable.run(work, "all", True, False, Path(temp) / "data")
                child.assert_not_called()
            self.assertEqual(json.loads((work / "run.json").read_text())["status"], "failed")

    def test_incomplete_inference_predictions_never_enter_scorer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            work = root / "fresh"
            def incomplete(command, log_path=None):
                table(work / "native", prepared()["selected_indices"], ["A"] * 4 + ["VIDEO_LOAD_FAILED"])
            with patch.object(launchable, "prepare", return_value=prepared()), \
                    patch.object(launchable, "endpoint"), \
                    patch.object(launchable, "run_command", side_effect=incomplete), \
                    patch.object(launchable, "score") as score:
                with self.assertRaisesRegex(RuntimeError, "failed/empty"):
                    launchable.run(work, "all", True, False, root / "data")
                score.assert_not_called()
            self.assertEqual(json.loads((work / "run.json").read_text())["status"], "failed")
            self.assertFalse((work / "scores").exists())

    def test_score_output_scope_and_source_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native = scorer_module()
            with patch.dict(sys.modules, {"vlmeval.perception_test_local": native}):
                with self.assertRaisesRegex(ValueError, "explicit"):
                    launchable.score(root, "subset", predictions=root / "predictions.tsv", score_dir=root / "new")
                with self.assertRaisesRegex(ValueError, "fresh --score-dir"):
                    launchable.score(root, "smoke", predictions=root / "predictions.tsv", score_dir=root)
                for source, status, dry_run in (("other-source", "inferred", False),
                                                (launchable.SOURCE, "failed", False),
                                                (launchable.SOURCE, "planned", True)):
                    inference_state(root, source=source, status=status, dry_run=dry_run)
                    with self.assertRaisesRegex(ValueError, "successful PerceptionTest"):
                        launchable.score(root, "smoke", work=root)
            native.score_predictions.assert_not_called()

    def test_changed_annotations_cuts_or_scope_fail_before_native_scoring(self):
        for change in ("scope", "dataset_revision", "manifest_sha256",
                       "annotation_manifest_sha256", "selected_indices"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                selection = prepared()
                updates = {change: "full" if change == "scope" else [9999]
                           if change == "selected_indices" else "changed"}
                inference_state(root, **updates)
                native = scorer_module()
                with patch.dict(sys.modules, {"vlmeval.perception_test_local": native}), \
                        patch.object(launchable, "prepare", return_value=selection):
                    with self.assertRaisesRegex(ValueError, "scope|changed"):
                        launchable.score(root / "data", "smoke", work=root)
                native.score_predictions.assert_not_called()
                self.assertFalse((root / "scores").exists())

    def test_prediction_artifact_cannot_escape_inference_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inference_state(root, predictions="../outside.tsv")
            native = scorer_module()
            with patch.dict(sys.modules, {"vlmeval.perception_test_local": native}):
                with self.assertRaisesRegex(ValueError, "within its inference work directory"):
                    launchable.score(root / "data", "smoke", work=root)
            native.score_predictions.assert_not_called()

    def test_failed_prediction_never_enters_native_scorer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = table(root / "native", [13404], ["VIDEO_LOAD_FAILED"])
            native = scorer_module()
            with patch.dict(sys.modules, {"vlmeval.perception_test_local": native}), \
                    patch.object(launchable, "prepare", return_value=prepared([13404], media=False)):
                with self.assertRaisesRegex(RuntimeError, "failed/empty"):
                    launchable.score(root / "data", "smoke", predictions=path, score_dir=root / "scores")
            native.score_predictions.assert_not_called()
            self.assertFalse((root / "scores").exists())

    def test_score_only_works_without_media_or_model_and_keeps_annotation_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selection = prepared([13404])
            path = table(root / "native", [13404], ["nan"])
            inference_state(root, selection, predictions=str(path.relative_to(root)))
            metadata = prepared([13404], media=False)
            def score_artifact(predictions, data_root, output_dir, smoke, nframe):
                self.assertEqual(predictions, path)
                self.assertTrue(smoke)
                self.assertFalse((data_root / "videos").exists())
                output_dir.mkdir()
                summary = {"status": "complete", "primary_metric": {
                    "name": "top1", "value": 0.0, "correct": 0, "total": 1},
                    "coverage": metadata["coverage"]}
                (output_dir / "summary.json").write_text(json.dumps(summary))
                return summary
            native = scorer_module(score_artifact)
            with patch.dict(sys.modules, {"vlmeval.perception_test_local": native}), \
                    patch.object(launchable, "prepare", return_value=metadata) as prep, \
                    patch.object(launchable, "endpoint") as endpoint, \
                    patch.object(launchable, "run_command") as child:
                summary = launchable.score(root / "metadata-only", "smoke", work=root)
                prep.assert_called_once_with(root / "metadata-only", smoke=True, require_media=False)
                endpoint.assert_not_called()
                child.assert_not_called()
            self.assertEqual(summary["primary_metric"]["total"], 1)
            self.assertEqual(json.loads((root / "run.json").read_text())["status"], "complete")
            controller = json.loads((root / "scores" / "controller.json").read_text())
            self.assertIn("ended_utc", controller["timing"])

    def test_scoring_execution_failure_is_retained_without_complete_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = table(root / "native", [13404])
            inference_state(root, prepared([13404]), predictions=str(path.relative_to(root)))
            def fail(predictions, data_root, output_dir, smoke, nframe):
                output_dir.mkdir()
                (output_dir / "diagnostic.txt").write_text("retain")
                raise RuntimeError("scoring failed")
            native = scorer_module(fail)
            with patch.dict(sys.modules, {"vlmeval.perception_test_local": native}), \
                    patch.object(launchable, "prepare", return_value=prepared([13404], media=False)):
                with self.assertRaisesRegex(RuntimeError, "scoring failed"):
                    launchable.score(root / "data", "smoke", work=root)
            state = json.loads((root / "run.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertIn("ended_utc", state["scoring_timing"])
            self.assertEqual((root / "scores" / "diagnostic.txt").read_text(), "retain")


if __name__ == "__main__":
    unittest.main()
