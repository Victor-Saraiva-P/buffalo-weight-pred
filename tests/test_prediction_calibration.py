from __future__ import annotations

import unittest

import numpy as np

from buffalo_weight.prediction_calibration import cross_fitted_calibration


class PredictionCalibrationTest(unittest.TestCase):
    def test_linear_calibration_corrects_known_cross_fold_scale(self) -> None:
        rows = [
            {"weight": str(weight), "y_pred": str(weight / 2), "fold": str(index % 2 + 1)}
            for index, weight in enumerate(range(100, 300, 20))
        ]

        calibrated = cross_fitted_calibration(rows, "linear")

        np.testing.assert_allclose(calibrated, np.asarray([float(row["weight"]) for row in rows]))


if __name__ == "__main__":
    unittest.main()
