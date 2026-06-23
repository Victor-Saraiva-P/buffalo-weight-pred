from __future__ import annotations

import unittest

from buffalo_weight.stability import evaluate_split_stability, split_random_states


class StabilityTest(unittest.TestCase):
    def test_evaluates_multiple_split_random_states(self) -> None:
        rows = []
        for index in range(40):
            rows.append(
                {
                    "file_name": f"mask-{index:03d}",
                    "farm": "A" if index % 2 == 0 else "B",
                    "tag": "train",
                    "weight": str(100 + index * 5),
                    "area": str(50 + index),
                    "perimeter": str(30 + index),
                    "solidity": "0.9",
                    "circularity": "0.8",
                    "equivalent_diameter": str(20 + index),
                    "hu_moment_1": "0.1",
                    "hu_moment_2": "0.01",
                }
            )

        fold_metrics, seed_summaries, overall, hard_examples = evaluate_split_stability(
            rows,
            ["area", "perimeter", "equivalent_diameter"],
            k=5,
            split_random_states=[0, 1, 2],
            n_estimators=5,
            training_random_state=42,
        )

        self.assertEqual(len(fold_metrics), 15)
        self.assertEqual(len(seed_summaries), 3)
        self.assertEqual(overall[0]["split_random_states"], "3")
        self.assertEqual(len(hard_examples), 40)
        self.assertEqual(set(fold_metrics[0]), {"split_random_state", "model", "fold", "mae", "rmse", "r2", "n_train", "n_validation"})

    def test_rejects_empty_seed_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "--seed-count must be at least 1"):
            split_random_states(start_seed=0, count=0)


if __name__ == "__main__":
    unittest.main()
