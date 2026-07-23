from __future__ import annotations

import unittest

from tools.calibrate_agglomeration_thresholds import PairTruth, evaluate, parse_truth


class AgglomerationCalibrationTests(unittest.TestCase):
    def test_truth_parser_accepts_review_labels(self):
        self.assertTrue(parse_truth("1"))
        self.assertTrue(parse_truth("团聚"))
        self.assertFalse(parse_truth("0"))
        self.assertFalse(parse_truth("非团聚"))
        self.assertIsNone(parse_truth(""))

    def test_threshold_metrics_reward_separation(self):
        rows = [
            PairTruth("a.csv", 0.10, 0.30, 0.08, True),
            PairTruth("a.csv", 0.20, 0.25, 0.06, True),
            PairTruth("a.csv", 0.10, 0.02, 0.00, False),
            PairTruth("a.csv", 0.80, 0.30, 0.00, False),
        ]
        metrics = evaluate(
            rows,
            tolerance_um=0.30,
            min_contact_ratio=0.12,
            min_overlap_ratio=0.03,
        )
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["tn"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
