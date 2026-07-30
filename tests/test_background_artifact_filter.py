from __future__ import annotations

import unittest

import cv2
import numpy as np

from metal_powder_sem_ai.postprocess import filter_false_background_masks
from metal_powder_sem_ai.segment import InstanceMask


def make_instance(
    local_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    internal_sides: tuple[str, ...],
    source: str = "tiled",
    tile_xyxy: tuple[int, int, int, int] | None = (0, 0, 90, 120),
) -> InstanceMask:
    contours, _ = cv2.findContours(
        (local_mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour = max(contours, key=cv2.contourArea).astype(np.int32)
    contour[:, 0, 0] += bbox[0]
    contour[:, 0, 1] += bbox[1]
    return InstanceMask(
        particle_id=1,
        mask=(local_mask > 0).astype(np.uint8),
        contour=contour,
        bbox_xyxy=bbox,
        score=0.95,
        source=source,
        tile_xyxy=tile_xyxy,
        internal_tile_edge_sides=internal_sides,
    )


class BackgroundArtifactFilterTests(unittest.TestCase):
    def test_dark_internal_tile_fragment_is_removed(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        image[24:28, 24:96] = 220
        mask = np.ones((60, 60), dtype=np.uint8)
        instance = make_instance(
            mask,
            (30, 30, 90, 90),
            internal_sides=("right",),
        )

        kept, info = filter_false_background_masks([instance], image)

        self.assertEqual(kept, [])
        self.assertEqual(info["background_filter_removed_count"], 1)

    def test_bright_particle_crossing_internal_tile_edge_is_kept(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        mask = np.zeros((60, 60), dtype=np.uint8)
        cv2.circle(mask, (30, 30), 26, 1, thickness=-1)
        image[30:90, 30:90][mask > 0] = 200
        instance = make_instance(
            mask,
            (30, 30, 90, 90),
            internal_sides=("right",),
        )

        kept, info = filter_false_background_masks([instance], image)

        self.assertEqual(len(kept), 1)
        self.assertEqual(info["background_filter_removed_count"], 0)

    def test_dark_mask_without_internal_tile_edge_is_not_removed(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        image[24:28, 24:96] = 220
        instance = make_instance(
            np.ones((60, 60), dtype=np.uint8),
            (30, 30, 90, 90),
            internal_sides=(),
            source="whole",
            tile_xyxy=None,
        )

        kept, info = filter_false_background_masks([instance], image)

        self.assertEqual(len(kept), 1)
        self.assertEqual(info["background_filter_removed_count"], 0)

    def test_dark_mask_near_internal_tile_edge_is_removed(self):
        image = np.full((100, 180, 3), 90, dtype=np.uint8)
        image[15:85, 20:98] = 25
        image[15:29, 20:98] = 200
        instance = make_instance(
            np.ones((70, 78), dtype=np.uint8),
            (20, 15, 98, 85),
            internal_sides=(),
            tile_xyxy=(0, 0, 100, 100),
        )

        kept, info = filter_false_background_masks([instance], image)

        self.assertEqual(kept, [])
        self.assertEqual(info["background_filter_removed_count"], 1)

    def test_dark_tiled_mask_far_from_internal_edge_is_kept(self):
        image = np.full((100, 180, 3), 180, dtype=np.uint8)
        image[20:80, 20:75] = 25
        instance = make_instance(
            np.ones((60, 55), dtype=np.uint8),
            (20, 20, 75, 80),
            internal_sides=(),
            tile_xyxy=(0, 0, 100, 100),
        )

        kept, info = filter_false_background_masks([instance], image)

        self.assertEqual(len(kept), 1)
        self.assertEqual(info["background_filter_removed_count"], 0)


if __name__ == "__main__":
    unittest.main()
