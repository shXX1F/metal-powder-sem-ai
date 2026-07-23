from __future__ import annotations

import unittest

from tools.prepare_agglomeration_review_set import is_accepted, stratified_sample


class AgglomerationReviewSetTests(unittest.TestCase):
    def test_acceptance_parser(self):
        self.assertTrue(is_accepted("True"))
        self.assertTrue(is_accepted("1"))
        self.assertFalse(is_accepted("False"))
        self.assertFalse(is_accepted(""))

    def test_sampling_is_reproducible_and_spreads_across_files(self):
        rows = []
        for file_index in range(4):
            for row_index in range(10):
                rows.append(
                    {
                        "source_csv": f"image_{file_index}.csv",
                        "source_row": str(row_index + 2),
                    }
                )
        first = stratified_sample(
            rows,
            target=12,
            max_per_file=3,
            seed=7,
        )
        second = stratified_sample(
            rows,
            target=12,
            max_per_file=3,
            seed=7,
        )
        self.assertEqual(first, second)
        counts = {}
        for row in first:
            key = row["source_csv"]
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(first), 12)
        self.assertEqual(set(counts.values()), {3})


if __name__ == "__main__":
    unittest.main()
