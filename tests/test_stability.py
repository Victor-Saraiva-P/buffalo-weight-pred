from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from buffalo_weight.models import ModelConfig
from buffalo_weight.stability import evaluate_split_stability, run_stability, split_random_states


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
            weight_category_count=4,
            split_random_states=[0, 1, 2],
            model_configs=[
                ModelConfig(
                    "random_forest_baseline",
                    "random_forest",
                    {"n_estimators": 5, "random_state": 42},
                )
            ],
        )

        self.assertEqual(len(fold_metrics), 15)
        self.assertEqual(len(seed_summaries), 3)
        self.assertEqual(overall[0]["split_random_states"], "3")
        self.assertEqual(len(hard_examples), 40)
        self.assertEqual(set(fold_metrics[0]), {"split_random_state", "model_config", "model", "fold", "mae", "rmse", "r2", "n_train", "n_validation"})

    def test_rejects_empty_seed_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "--seed-count must be at least 1"):
            split_random_states(start_seed=0, count=0)

    def test_stability_uses_configured_weight_category_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features_path = root / "features.csv"
            config_path = root / "config.yaml"
            output_dir = root / "diagnostics"
            fieldnames = [
                "file_name",
                "farm",
                "weight",
                "tag",
                "area",
                "perimeter",
                "solidity",
                "circularity",
                "equivalent_diameter",
                "hu_moment_1",
                "hu_moment_2",
            ]
            with features_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for index in range(40):
                    writer.writerow(
                        {
                            "file_name": f"mask-{index:03d}",
                            "farm": "A",
                            "weight": str(100 + index * 5),
                            "tag": "train",
                            "area": str(50 + index),
                            "perimeter": str(30 + index),
                            "solidity": "0.9",
                            "circularity": "0.8",
                            "equivalent_diameter": str(20 + index),
                            "hu_moment_1": "0.1",
                            "hu_moment_2": "0.01",
                        }
                    )
            config_path.write_text(
                "\n".join(
                    [
                        "output:",
                        f"  features_index_path: {features_path}",
                        "split:",
                        "  k: 5",
                        "  weight_category_count: 8",
                        "training:",
                        "  model_configs:",
                        "    random_forest_baseline:",
                        "      model: random_forest",
                        "      params:",
                        "        n_estimators: 5",
                        "        random_state: 42",
                        "  feature_columns:",
                        "    - area",
                        "    - perimeter",
                        "    - equivalent_diameter",
                    ]
                )
            )

            run_stability(config_path, start_seed=0, seed_count=1, output_dir=output_dir)

            with (output_dir / "random_forest_baseline" / "hard_examples.csv").open() as file:
                hard_examples = list(csv.DictReader(file))
            self.assertEqual(
                {row["weight_category"] for row in hard_examples},
                {f"B{index}" for index in range(1, 9)},
            )


if __name__ == "__main__":
    unittest.main()
