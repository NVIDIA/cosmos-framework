# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller behavior regressions; synthetic artifacts, no GPU or network."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import launchable


def prepared(indices=(3400, 3401, 3402, 3403, 3404)):
    return {"dataset_revision": "pinned-test-revision", "manifest_sha256": "trusted-hash",
            "selected_indices": list(indices), "coverage": {
                "expected_rows": 4000, "selected_rows": len(indices),
                "available_rows": len(indices), "expected_tasks": 20,
                "selected_tasks": 1, "subset": len(indices) != 4000}}


def table(work, indices, values=None, stamp="T20261029"):
    output = work / "cosmos" / stamp
    output.mkdir(parents=True)
    path = output / "cosmos_MVBench.tsv"
    values = values if values is not None else ["A"] * len(indices)
    path.write_text("index\tprediction\n" + "".join(
        f"{index}\t{value}\n" for index, value in zip(indices, values)))
    return path


class ArtifactTests(unittest.TestCase):
    def test_existing_output_rejected_before_network_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(launchable, "prepare") as prep:
            with self.assertRaisesRegex(ValueError, "fresh --work-dir"):
                launchable.run(Path(temp), "infer", True, False, Path(temp) / "data")
            prep.assert_not_called()

    def test_nonpositive_parameters_rejected_before_output(self):
        for workers, frames in ((0, 16), (-1, 16), (1, 0), (1, -1)):
            with self.subTest(workers=workers, frames=frames), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "fresh"
                with patch.object(launchable, "prepare") as prep:
                    with self.assertRaisesRegex(ValueError, "positive"):
                        launchable.run(output, "infer", False, False, Path(temp) / "missing",
                                       api_nproc=workers, nframe=frames)
                    prep.assert_not_called()
                self.assertFalse(output.exists())

    def test_coverage_failure_precedes_endpoint_and_output(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / "fresh"
            with patch.object(launchable, "prepare", side_effect=ValueError("missing NTU")), \
                    patch.object(launchable, "endpoint") as endpoint:
                with self.assertRaisesRegex(ValueError, "missing NTU"):
                    launchable.run(work, "all", False, False, Path(temp) / "data")
                endpoint.assert_not_called()
            self.assertFalse(work.exists())

    def test_predictions_reject_empty_and_failure_markers(self):
        for value in ("   ", "VIDEO_LOAD_FAILED", "prefix VIDEO_LOAD_FAILED",
                      "Failed to obtain answer via API."):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp:
                path = table(Path(temp), [3400], [value])
                with self.assertRaisesRegex(RuntimeError, "failed/empty"):
                    launchable.verify_predictions(path, [3400])

    def test_selected_identities_exact_even_when_row_count_matches(self):
        for indices in ([3400, 3400], [3400, 9999], [3400], [3400, 3401, 3402]):
            with self.subTest(indices=indices), tempfile.TemporaryDirectory() as temp:
                path = table(Path(temp), indices)
                with self.assertRaisesRegex(RuntimeError, "identities|expected 2 predictions"):
                    launchable.verify_predictions(path, [3400, 3401])


    def test_multiple_dated_prediction_tables_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            table(work, [3400], stamp="T20261028")
            table(work, [3400], stamp="T20261029")
            with self.assertRaisesRegex(RuntimeError, "expected one prediction table, found 2"):
                launchable.verify_predictions(work, [3400])

    def test_inference_failure_marks_state_and_keeps_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            work = root / "fresh"
            def fail(command, log_path=None):
                Path(log_path).write_text("private diagnostic\n")
                raise subprocess.CalledProcessError(3, command)
            with patch.object(launchable, "prepare", return_value=prepared()), \
                    patch.object(launchable, "endpoint"), \
                    patch.object(launchable, "run_command", side_effect=fail):
                with self.assertRaises(subprocess.CalledProcessError):
                    launchable.run(work, "all", True, False, root / "data")
            self.assertEqual(json.loads((work / "run.json").read_text())["status"], "failed")
            self.assertEqual((work / "run.log").read_text(), "private diagnostic\n")
            self.assertTrue((work / "config.json").is_file())


    def test_score_output_and_scope_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native = types.ModuleType("vlmeval.mvbench_local")
            native.score_predictions = unittest.mock.Mock()
            with patch.dict(sys.modules, {"vlmeval.mvbench_local": native}):
                with self.assertRaisesRegex(ValueError, "explicit"):
                    launchable.score(root, "subset", predictions=root / "predictions.tsv", score_dir=root / "new")
                with self.assertRaisesRegex(ValueError, "fresh --score-dir"):
                    launchable.score(root, "smoke", predictions=root / "predictions.tsv", score_dir=root)
            native.score_predictions.assert_not_called()

    def test_changed_annotations_or_scope_fail_before_native_scoring(self):
        for change in ("scope", "manifest_sha256", "selected_indices"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                selection = prepared()
                state = {**selection, "source": "mvbench-local", "status": "inferred",
                         "scope": "smoke", "nframe": 16, "predictions": "native.tsv"}
                if change == "scope":
                    state["scope"] = "full"
                elif change == "manifest_sha256":
                    state[change] = "different-hash"
                else:
                    state[change] = [9999]
                (root / "run.json").write_text(json.dumps(state))
                native = types.ModuleType("vlmeval.mvbench_local")
                native.score_predictions = unittest.mock.Mock()
                with patch.dict(sys.modules, {"vlmeval.mvbench_local": native}), \
                        patch.object(launchable, "prepare", return_value=selection):
                    with self.assertRaisesRegex(ValueError, "scope|changed"):
                        launchable.score(root / "data", "smoke", work=root)
                native.score_predictions.assert_not_called()
                self.assertFalse((root / "scores").exists())

    def test_failed_prediction_never_enters_native_scorer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = table(root / "native", [3400], ["VIDEO_LOAD_FAILED"])
            native = types.ModuleType("vlmeval.mvbench_local")
            native.score_predictions = unittest.mock.Mock()
            with patch.dict(sys.modules, {"vlmeval.mvbench_local": native}), \
                    patch.object(launchable, "prepare", return_value=prepared([3400])):
                with self.assertRaisesRegex(RuntimeError, "failed/empty"):
                    launchable.score(root / "data", "smoke", predictions=path, score_dir=root / "scores")
            native.score_predictions.assert_not_called()
            self.assertFalse((root / "scores").exists())

    def test_streamed_child_output_and_failure_log_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run.log"
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                with self.assertRaises(subprocess.CalledProcessError):
                    launchable.run_command([sys.executable, "-u", "-c",
                                            "print('first', flush=True); raise SystemExit(7)"], log_path=path)
            self.assertEqual(captured.getvalue(), "first\n")
            self.assertEqual(path.read_text(), "first\n")


if __name__ == "__main__":
    unittest.main()
