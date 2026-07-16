from __future__ import annotations

import unittest

import numpy as np

from buffalo_weight.diagnostic_robustness import perturb_masks


class DiagnosticRobustnessTest(unittest.TestCase):
    def test_erosion_and_dilation_change_foreground_area(self) -> None:
        masks = np.zeros((1, 9, 9), dtype=np.float32)
        masks[:, 2:7, 2:7] = 1

        eroded = perturb_masks(masks, "erosion_1")
        dilated = perturb_masks(masks, "dilation_1")

        self.assertLess(float(eroded.sum()), float(masks.sum()))
        self.assertGreater(float(dilated.sum()), float(masks.sum()))


if __name__ == "__main__":
    unittest.main()
