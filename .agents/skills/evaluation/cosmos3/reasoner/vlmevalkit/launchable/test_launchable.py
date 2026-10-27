# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact safety regressions; no GPU, credentials, or network required."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import launchable


class ArtifactTests(unittest.TestCase):
    def test_existing_output_is_rejected_before_network_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "fresh --work-dir"):
                launchable.run(["VANTAGE_VQA"], Path(temp), "infer", True, False, Path(temp) / "assets")

    def test_predictions_reject_empty_and_failure_markers(self):
        for value in ("   ", "VIDEO_LOAD_FAILED", "prefix VIDEO_LOAD_FAILED",
                      "Failed to obtain answer via API.", "null"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp:
                work = Path(temp)
                output = work / "cosmos" / "T20261023"
                output.mkdir(parents=True)
                (output / "cosmos_VANTAGE_VQA.tsv").write_text(
                    "index\tprediction\n0\t" + value + "\n")
                with self.assertRaisesRegex(RuntimeError, "failed/empty"):
                    launchable.verify_predictions(work, "VANTAGE_VQA")

    def test_smoke_requires_exactly_five_predictions(self):
        for count in (4, 5, 6):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temp:
                work = Path(temp)
                output = work / "cosmos" / "T20261023"
                output.mkdir(parents=True)
                (output / "cosmos_VANTAGE_VQA.tsv").write_text(
                    "index\tprediction\n" + "".join(f"{i}\tA\n" for i in range(count)))
                if count == 5:
                    with contextlib.redirect_stdout(io.StringIO()):
                        launchable.verify_predictions(work, "VANTAGE_VQA", expected_rows=5)
                else:
                    with self.assertRaisesRegex(RuntimeError, "expected 5 predictions"):
                        launchable.verify_predictions(work, "VANTAGE_VQA", expected_rows=5)

    def test_multiple_dated_prediction_tables_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            for stamp in ("T20261023", "T20261024"):
                output = work / "cosmos" / stamp
                output.mkdir(parents=True)
                (output / "cosmos_VANTAGE_VQA.tsv").write_text("index\tprediction\n0\tA\n")
            with self.assertRaisesRegex(RuntimeError, "expected one prediction table, found 2"):
                launchable.verify_predictions(work, "VANTAGE_VQA")


if __name__ == "__main__":
    unittest.main()
