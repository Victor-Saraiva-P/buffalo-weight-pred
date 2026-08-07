"""Tests for nested stratified subset generation for controlled learning curves."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from buffalo_weight.diagnostic_learning_subsets import generate_nested_subsets


@dataclass(frozen=True)
class DummySample:
    file_name: str
    weight_category: str
    fold: int


class DiagnosticLearningSubsetsTest(unittest.TestCase):
    def setUp(self) -> None:
        # Create 100 samples across 5 folds with 2 weight categories
        self.samples: list[DummySample] = []
        for i in range(100):
            fold = (i % 5) + 1
            cat = "B1" if i % 2 == 0 else "B10"
            self.samples.append(DummySample(file_name=f"sample_{i:03d}.png", weight_category=cat, fold=fold))

    def test_outer_fold_isolation(self) -> None:
        """Held-out fold samples must never appear in any training subset."""
        for fold in range(1, 6):
            subsets = generate_nested_subsets(self.samples, outer_fold=fold, seed=45)
            for frac, sub_samples in subsets.items():
                for sample in sub_samples:
                    self.assertNotEqual(
                        sample.fold,
                        fold,
                        f"sample {sample.file_name} from fold {fold} appeared in fraction {frac}",
                    )

    def test_subset_nesting(self) -> None:
        """50% subset must be a subset of 75%, and 75% a subset of 100%."""
        subsets = generate_nested_subsets(self.samples, outer_fold=1, seed=45)
        names_50 = {s.file_name for s in subsets[0.50]}
        names_75 = {s.file_name for s in subsets[0.75]}
        names_100 = {s.file_name for s in subsets[1.00]}

        self.assertTrue(names_50.issubset(names_75), "50% subset is not contained in 75% subset")
        self.assertTrue(names_75.issubset(names_100), "75% subset is not contained in 100% subset")
        self.assertEqual(len(subsets[0.50]), round(len(subsets[1.00]) * 0.50))
        self.assertEqual(len(subsets[0.75]), round(len(subsets[1.00]) * 0.75))

    def test_subset_stratification(self) -> None:
        """Weight category proportions should be preserved across subsets."""
        subsets = generate_nested_subsets(self.samples, outer_fold=1, seed=45)
        for frac, sub in subsets.items():
            b1_count = sum(1 for s in sub if s.weight_category == "B1")
            b10_count = sum(1 for s in sub if s.weight_category == "B10")
            ratio = b1_count / len(sub)
            self.assertAlmostEqual(ratio, 0.5, delta=0.1, msg=f"fraction {frac} unstratified")

    def test_repeatability_and_determinism(self) -> None:
        """Repeated generation with seed 45 must produce identical subsets."""
        run1 = generate_nested_subsets(self.samples, outer_fold=2, seed=45)
        run2 = generate_nested_subsets(self.samples, outer_fold=2, seed=45)
        for frac in (0.50, 0.75, 1.00):
            names1 = [s.file_name for s in run1[frac]]
            names2 = [s.file_name for s in run2[frac]]
            self.assertEqual(names1, names2)


if __name__ == "__main__":
    unittest.main()
