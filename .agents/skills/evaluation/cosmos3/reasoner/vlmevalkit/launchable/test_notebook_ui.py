# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prevent mislabeled or incomplete cross-benchmark result cards."""
from pathlib import Path
import unittest

from notebook_ui import show_results


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


if __name__ == '__main__':
    unittest.main()
