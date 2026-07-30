from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from tools.calibrate_agglomeration_thresholds import (
    PairTruth,
    evaluate,
    load_truth,
    parse_truth,
)


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

    def test_pixel_quantization_matches_production_gap_rule(self):
        row = PairTruth(
            "a.csv",
            gap_um=0.42,
            contact_ratio=0.20,
            overlap_ratio=0.00,
            truth=True,
            gap_px=2.9,
            pixel_size_um=0.14,
            evidence_tolerance_px=3,
        )
        metrics = evaluate(
            [row],
            tolerance_um=0.30,
            min_contact_ratio=0.12,
            min_overlap_ratio=0.03,
        )
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["contact_profile_exact_fraction"], 1.0)

    def test_duplicate_review_rows_are_counted_once(self):
        fieldnames = [
            "source_csv",
            "source_row",
            "manual_truth",
            "gap_um",
            "gap_px",
            "pixel_size_um",
            "tolerance_px",
            "contact_ratio",
            "overlap_ratio",
        ]
        row = {
            "source_csv": "98-3_merged.agglomerate_pairs.csv",
            "source_row": "715",
            "manual_truth": "1",
            "gap_um": "0.20",
            "gap_px": "2",
            "pixel_size_um": "0.10",
            "tolerance_px": "3",
            "contact_ratio": "0.10",
            "overlap_ratio": "0.00",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for name in ("review_a.csv", "review_b.csv"):
                path = Path(temp_dir) / name
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(row)
                paths.append(path)

            loaded, stats = load_truth(paths)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(stats["duplicate_pairs_removed"], 1)
        self.assertEqual(stats["ignored_unlabeled_or_invalid_rows"], 0)


if __name__ == "__main__":
    unittest.main()
