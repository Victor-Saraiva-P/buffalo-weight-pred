from __future__ import annotations

import unittest

from buffalo_weight.feature_analysis import classical_model_configs
from buffalo_weight.feature_analysis_core import permutation_seed, validate_feature_values
from buffalo_weight.feature_analysis_reports import redundant_pairs, summarize_by_model


class FeatureAnalysisTest(unittest.TestCase):
    def test_rejects_negative_geometric_feature(self) -> None:
        rows = [{"file_name": "mask-001", "weight": "100", "area": "-1"}]

        with self.assertRaisesRegex(ValueError, "expected a non-negative number"):
            validate_feature_values(rows, ["area"])

    def test_allows_negative_hu_moment_feature(self) -> None:
        rows = [{"file_name": "mask-001", "weight": "100", "hu_moment_1": "-0.1"}]

        validate_feature_values(rows, ["hu_moment_1"])

    def test_permutation_seed_is_deterministic_by_feature(self) -> None:
        self.assertEqual(permutation_seed(3, 2, "area"), permutation_seed(3, 2, "area"))
        self.assertNotEqual(permutation_seed(3, 2, "area"), permutation_seed(3, 2, "perimeter"))

    def test_rejects_mask_model_config(self) -> None:
        config = {
            "model_configs": {
                "cnn_mask_baseline": {
                    "model": "cnn_mask",
                    "params": {
                        "epochs": 1,
                        "batch_size": 1,
                        "learning_rate": 0.01,
                        "image_size": 8,
                        "random_state": 42,
                    },
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "only supports classical models"):
            classical_model_configs(config)

    def test_summarizes_feature_impacts_by_model(self) -> None:
        rows = [
            metric("random_forest_baseline", "all_features", "", "10"),
            metric("random_forest_baseline", "single_feature", "area", "12"),
            metric("random_forest_baseline", "without_feature", "area", "15"),
            metric("random_forest_baseline", "permuted_feature", "area", "14"),
            metric("random_forest_baseline", "mean_baseline", "", "20"),
            metric("random_forest_baseline", "area_baseline", "area", "12"),
        ]

        summary = summarize_by_model(rows, ["area"])

        self.assertEqual(summary[0]["removal_impact_mae"], "5")
        self.assertEqual(summary[0]["permutation_impact_mae"], "4")

    def test_reports_highly_redundant_pairs(self) -> None:
        rows = [
            {"file_name": "a", "weight": "100", "area": "1", "bbox_area": "10", "perimeter": "9"},
            {"file_name": "b", "weight": "110", "area": "2", "bbox_area": "20", "perimeter": "7"},
            {"file_name": "c", "weight": "120", "area": "3", "bbox_area": "30", "perimeter": "8"},
        ]

        pairs = redundant_pairs(rows, ["area", "bbox_area", "perimeter"])

        self.assertIn({"feature_a": "area", "feature_b": "bbox_area", "spearman": "1", "pearson": "1"}, pairs)


def metric(model_config: str, scenario: str, feature: str, mae: str) -> dict[str, str]:
    return {
        "model_config": model_config,
        "model": "random_forest",
        "scenario": scenario,
        "feature": feature,
        "split_random_state": "0",
        "fold": "1",
        "mae": mae,
        "rmse": mae,
        "r2": "0",
        "n_train": "2",
        "n_validation": "1",
    }


if __name__ == "__main__":
    unittest.main()
