# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prevent mislabeled or incomplete cross-benchmark result cards."""
from pathlib import Path
import unittest

from notebook_ui import in_expected_range, show_results


class ResultIdentityTests(unittest.TestCase):
    def test_rejects_another_benchmarks_artifact_before_display(self):
        with self.assertRaises(ValueError):
            show_results({'benchmark': 'mmsivideo-bench', 'scope': 'full'},
                         'videophy2', 'full', Path('/unused'))

    def test_rejects_smoke_artifact_as_a_full_benchmark(self):
        with self.assertRaises(ValueError):
            show_results({'benchmark': 'perception-test', 'scope': 'smoke'},
                         'perception-test', 'full', Path('/unused'))

    def test_rejects_failed_run_even_with_a_numeric_score(self):
        with self.assertRaises(ValueError):
            show_results({'benchmark': 'videophy2', 'scope': 'full', 'status': 'failed',
                          'primary_metric': {'value': 48.13, 'unit': 'pearson_x100'}},
                         'videophy2', 'full', Path('/unused'))

    def test_incomplete_coverage_cannot_be_presented_as_complete(self):
        with self.assertRaises(ValueError):
            show_results({'benchmark': 'mmsivideo-bench', 'scope': 'full',
                          'status': 'completed', 'coverage': {'expected': 1106, 'scored': 1105},
                          'primary_metric': {'value': 27.94, 'unit': 'percent'}},
                         'mmsivideo-bench', 'full', Path('/unused'))


class ExpectedRangeTests(unittest.TestCase):
    FULL = {'expected': 1106, 'scored': 1106}
    NANO = 'nvidia/Cosmos3-Nano'

    def test_band_is_one_point_either_side_inclusive(self):
        self.assertTrue(in_expected_range('mmsivideo-bench', 100 * 298 / 1106, self.FULL, self.NANO))
        self.assertTrue(in_expected_range('mmsivideo-bench', 100 * 320 / 1106, self.FULL, self.NANO))
        self.assertFalse(in_expected_range('mmsivideo-bench', 100 * 297 / 1106, self.FULL, self.NANO))
        self.assertFalse(in_expected_range('mmsivideo-bench', 100 * 321 / 1106, self.FULL, self.NANO))

    def test_partial_or_resized_run_is_never_in_range(self):
        self.assertFalse(in_expected_range('mmsivideo-bench', 27.94, {'expected': 1000, 'scored': 1000}, self.NANO))

    def test_other_models_have_no_expected_range(self):
        self.assertIsNone(in_expected_range('mmsivideo-bench', 31.0, self.FULL, 'nvidia/Cosmos3.5-Ultra-Reasoner'))

    def test_benchmarks_without_an_approved_band_report_none(self):
        self.assertIsNone(in_expected_range('videophy2', 48.13, {'expected': 3397, 'scored': 3397}, self.NANO))


if __name__ == '__main__':
    unittest.main()
