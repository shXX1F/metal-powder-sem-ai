from __future__ import annotations

import unittest

from tools.import_agglomeration_review_xlsx import (
    batch_id_from_source_csv,
    normalize_label,
    normalize_source_row,
    review_key,
)


class AgglomerationReviewImportTests(unittest.TestCase):
    def test_excel_numeric_identifiers_are_normalized(self):
        self.assertEqual(normalize_source_row(715.0), "715")
        self.assertEqual(
            review_key(
                {
                    "source_csv": r"C:\review\98-3_merged.agglomerate_pairs.csv",
                    "source_row": 715.0,
                }
            ),
            "98-3_merged.agglomerate_pairs.csv#715",
        )

    def test_manual_truth_accepts_invalid_mask_marker(self):
        self.assertEqual(normalize_label(1.0), "1")
        self.assertEqual(normalize_label(0), "0")
        self.assertEqual(normalize_label(-1), "-1")
        self.assertEqual(normalize_label(None), "")
        with self.assertRaises(ValueError):
            normalize_label("maybe")

    def test_batch_id_is_recovered_from_source_csv(self):
        self.assertEqual(
            batch_id_from_source_csv("98-3_merged.agglomerate_pairs.csv"),
            "98-3",
        )


if __name__ == "__main__":
    unittest.main()
