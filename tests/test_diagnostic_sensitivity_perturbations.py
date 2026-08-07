"""Tests for mask perturbation primitives.

Covers: disk construction, rounding, border handling, topology validation.
"""

from __future__ import annotations

import unittest

import numpy as np

from buffalo_weight.diagnostic_sensitivity_perturbations import (
    _build_disk,
    _round_half_up,
    compute_shift_pixels,
    count_four_neighbor_components,
    euclidean_disk,
    has_valid_topology,
    perturb_contraction,
    perturb_expansion,
    perturb_scale_grow,
    perturb_scale_shrink,
    perturb_shift,
)


class EuclideanDiskTest(unittest.TestCase):
    def test_disk_radius_1_is_3x3_cross(self) -> None:
        disk = _build_disk(1)
        self.assertEqual(disk.shape, (3, 3))
        # Center and orthogonal neighbors are 1; corners are 0
        expected = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(disk, expected)

    def test_disk_radius_2_is_5x5(self) -> None:
        disk = _build_disk(2)
        self.assertEqual(disk.shape, (5, 5))
        # Corners at distance sqrt(8) > 2, so excluded
        self.assertEqual(disk[0, 0], 0)
        self.assertEqual(disk[2, 2], 1)  # center

    def test_disk_is_symmetric(self) -> None:
        disk = _build_disk(5)
        np.testing.assert_array_equal(disk, disk[::-1, :])
        np.testing.assert_array_equal(disk, disk[:, ::-1])


class RoundHalfUpTest(unittest.TestCase):
    def test_exact_integer(self) -> None:
        self.assertEqual(_round_half_up(3.0), 3)

    def test_below_half(self) -> None:
        self.assertEqual(_round_half_up(2.4), 2)

    def test_half_rounds_up(self) -> None:
        self.assertEqual(_round_half_up(2.5), 3)

    def test_above_half(self) -> None:
        self.assertEqual(_round_half_up(2.7), 3)


class EuclideanDiskCanonicalConversionTest(unittest.TestCase):
    def test_same_scale_preserves_radius(self) -> None:
        disk = euclidean_disk(5, 1024, 1024)
        self.assertEqual(disk.shape, (11, 11))

    def test_half_scale_halves_radius(self) -> None:
        # radius_canonical=5, 1024->512 => ratio=0.5, 5*0.5=2.5 -> rounds up to 3
        disk = euclidean_disk(5, 1024, 512)
        self.assertEqual(disk.shape, (7, 7))  # 2*3+1

    def test_minimum_radius_is_1(self) -> None:
        # Very small original image
        disk = euclidean_disk(5, 1024, 10)
        # 5 * 10/1024 = 0.049, rounds to 0, clamped to 1
        self.assertEqual(disk.shape, (3, 3))


class ScalePerturbationTest(unittest.TestCase):
    def test_shrink_reduces_foreground(self) -> None:
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[30:70, 30:70] = 1.0
        shrunk = perturb_scale_shrink(mask, 0.05)
        self.assertLess(np.sum(shrunk > 0), np.sum(mask > 0))

    def test_grow_increases_foreground(self) -> None:
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[30:70, 30:70] = 1.0
        grown = perturb_scale_grow(mask, 0.05)
        self.assertGreater(np.sum(grown > 0), np.sum(mask > 0))

    def test_empty_mask_unchanged(self) -> None:
        mask = np.zeros((50, 50), dtype=np.float32)
        result = perturb_scale_shrink(mask)
        self.assertEqual(np.sum(result), 0)


class ShiftPerturbationTest(unittest.TestCase):
    def test_shift_without_cutting(self) -> None:
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:80, 20:80] = 1.0
        shifted, cuts = perturb_shift(mask, -5, 0)
        self.assertFalse(cuts)
        self.assertEqual(np.sum(shifted > 0), np.sum(mask > 0))

    def test_shift_cutting_foreground(self) -> None:
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[0:10, 10:40] = 1.0  # foreground at top edge
        _, cuts = perturb_shift(mask, -5, 0)
        self.assertTrue(cuts)

    def test_compute_shift_pixels(self) -> None:
        mask = np.zeros((100, 80), dtype=np.float32)
        shift_px = compute_shift_pixels(mask, 0.05)
        self.assertEqual(shift_px, 5)  # 100 * 0.05 = 5


class MorphologyPerturbationTest(unittest.TestCase):
    def test_contraction_reduces_foreground(self) -> None:
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[10:40, 10:40] = 1.0
        se = _build_disk(2)
        contracted = perturb_contraction(mask, se)
        self.assertLess(np.sum(contracted), np.sum(mask > 0))

    def test_expansion_increases_foreground(self) -> None:
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[10:40, 10:40] = 1.0
        se = _build_disk(2)
        expanded = perturb_expansion(mask, se)
        self.assertGreater(np.sum(expanded), np.sum(mask > 0))


class TopologyTest(unittest.TestCase):
    def test_single_component_preserved(self) -> None:
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[10:40, 10:40] = 1.0
        se = _build_disk(1)
        contracted = perturb_contraction(mask, se)
        self.assertTrue(has_valid_topology(mask, contracted))

    def test_erosion_splitting_component_detected(self) -> None:
        # Two blobs connected by a thin bridge — erosion should split them
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[5:20, 5:20] = 1.0
        mask[30:45, 30:45] = 1.0
        mask[20, 20:30] = 1.0  # thin bridge
        se = _build_disk(2)
        contracted = perturb_contraction(mask, se)
        # The bridge should be eroded, splitting the component
        self.assertFalse(has_valid_topology(mask, contracted))

    def test_count_four_neighbor_components(self) -> None:
        mask = np.zeros((20, 20), dtype=np.float32)
        mask[2:8, 2:8] = 1.0
        mask[12:18, 12:18] = 1.0
        self.assertEqual(count_four_neighbor_components(mask), 2)

    def test_diagonal_neighbors_not_connected(self) -> None:
        # Two pixels touching diagonally — should be 2 components with 4-connectivity
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[3, 3] = 1.0
        mask[4, 4] = 1.0
        self.assertEqual(count_four_neighbor_components(mask), 2)


if __name__ == "__main__":
    unittest.main()
