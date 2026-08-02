from __future__ import annotations

import csv
import io
from pathlib import Path
import tempfile
import unittest

from buffalo_weight.category_comparison import main, run_category_comparison
from buffalo_weight.environment_contract import ComputeEnvironment
from tests.fake_setup_services import FakeRuntimeProbe


NEURAL_CATEGORY_CONFIG: dict[str, object] = {
    "output": {"features_index_path": "must-not-be-read.csv"},
    "split": {"k": 5},
    "training": {
        "feature_columns": ["area"],
        "model_configs": {
            "cnn_mask_baseline": {
                "model": "cnn_mask",
                "params": {
                    "batch_size": 8,
                    "epochs": 1,
                    "image_size": 64,
                    "learning_rate": 0.001,
                    "random_state": 42,
                },
            }
        },
    },
}


class FakeCategoryConfigLoader:
    def __call__(self, path: Path) -> dict[str, object]:
        """Return a neural config whose feature path must never be read."""

        return NEURAL_CATEGORY_CONFIG


class CategoryComparisonTest(unittest.TestCase):
    def test_neural_cli_rejects_missing_cuda_before_reading_features(self) -> None:
        runtime = FakeRuntimeProbe(
            compute=ComputeEnvironment(None, None, "13.0", "590.00")
        )
        stderr = io.StringIO()

        result = main(
            ["--config", "neural.yaml"], runtime_probe=runtime,
            config_loader=FakeCategoryConfigLoader(), stderr=stderr,
        )

        self.assertEqual(result, 1)
        self.assertIn("available CUDA GPU", stderr.getvalue())

    def test_compares_configured_weight_category_counts(self) -> None:
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
                            "farm": "A" if index % 2 == 0 else "B",
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
                        "training:",
                        "  model_configs:",
                        "    random_forest_baseline:",
                        "      model: random_forest",
                        "      params:",
                        "        n_estimators: 5",
                        "        random_state: 42",
                        "    xgboost_baseline:",
                        "      model: xgboost",
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

            run_category_comparison(
                config_path,
                category_counts=[4, 8],
                start_seed=0,
                seed_count=2,
                output_dir=output_dir,
            )

            with (output_dir / "model_comparison.csv").open() as file:
                overall = list(csv.DictReader(file))
            with (output_dir / "random_forest_baseline" / "fold_metrics.csv").open() as file:
                fold_metrics = list(csv.DictReader(file))
            with (output_dir / "split_balance.csv").open() as file:
                split_balance = list(csv.DictReader(file))

            self.assertEqual(
                {row["weight_category_count"] for row in overall}, {"4", "8"}
            )
            self.assertEqual(
                {row["model_config"] for row in overall},
                {"random_forest_baseline", "xgboost_baseline"},
            )
            self.assertEqual(len(overall), 4)
            self.assertEqual(len(fold_metrics), 20)
            self.assertEqual(
                {row["weight_category_count"] for row in split_balance}, {"4", "8"}
            )
            self.assertEqual(set(split_balance[0]), {"weight_category_count", "split_random_state", "fold", "weight_category", "n_validation", "weight_min", "weight_median", "weight_max"})
            self.assertTrue((output_dir / "model_comparison.png").exists())
            self.assertTrue((output_dir / "random_forest_baseline" / "mae.png").exists())
            self.assertTrue((output_dir / "random_forest_baseline" / "seed_variation.png").exists())
            self.assertTrue((output_dir / "weight_scatter_c4.png").exists())
            self.assertTrue((output_dir / "weight_heatmap_c8.png").exists())


if __name__ == "__main__":
    unittest.main()
