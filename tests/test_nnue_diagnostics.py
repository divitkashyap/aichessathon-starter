import unittest

import numpy as np

from tools.validate_nnue import summarize


class DiagnosticTests(unittest.TestCase):
    def test_disaggregates_extremes_and_endgames(self):
        result = summarize(
            np.array([0.0, 100.0, 9900.0]), np.array([2000.0, 120.0, 5000.0]), np.array([5, 30, 5])
        )
        self.assertEqual(result["all"]["count"], 3)
        self.assertEqual(result["ordinary_8orless_pieces"]["mae_cp"], 2000.0)
        self.assertEqual(result["ordinary_17plus_pieces"]["mae_cp"], 20.0)
        self.assertEqual(result["ordinary_abs_target_le_300"]["count"], 2)
        self.assertEqual(result["extreme_abs_target_gt_1000"]["count"], 1)

    def test_empty_groups_are_omitted(self):
        result = summarize(np.array([50.0]), np.array([60.0]), np.array([32]))
        self.assertNotIn("ordinary_8orless_pieces", result)
        self.assertEqual(result["all"]["zero_baseline_mae_cp"], 50.0)
