from __future__ import annotations

import unittest

import numpy as np

from buffalo_weight.target_transform import (
    inverse_target,
    inverse_target_power,
    transform_target,
    transform_target_power,
)


class TargetTransformTest(unittest.TestCase):
    def test_round_trips_supported_allometric_transforms(self) -> None:
        weights = np.asarray([64.0, 125.0, 216.0])

        for transform in ("identity", "log", "cube_root"):
            restored = inverse_target(transform_target(weights, transform), transform)
            np.testing.assert_allclose(restored, weights)

    def test_rejects_unknown_transform(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected one of"):
            transform_target(np.asarray([100.0]), "unknown")

    def test_round_trips_continuous_target_power(self) -> None:
        weights = np.asarray([64.0, 125.0, 216.0])

        restored = inverse_target_power(transform_target_power(weights, 0.25), 0.25)

        np.testing.assert_allclose(restored, weights)


if __name__ == "__main__":
    unittest.main()
