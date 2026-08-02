from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from buffalo_weight.pure_geometry_evaluation import PURE_GEOMETRY_FEATURES, NestedEvaluation
from buffalo_weight.train_pure_geometry import main, train_pure_geometry


class FakeNestedEvaluator:
    def __call__(self, rows: list[dict[str, str]], random_state: int, inner_k: int) -> NestedEvaluation:
        return NestedEvaluation([], [{"model": "ridge", "fold": "1"}], [], [])


class FakeGeometryReporter:
    def __call__(
        self, evaluation: NestedEvaluation, rows: list[dict[str, str]], output_dir: Path
    ) -> list[dict[str, str]]:
        return [{"model": "ridge", "mae_kg": "1", "r2": "0.9"}]


class TrainPureGeometryTest(unittest.TestCase):
    def test_runs_injected_evaluator_and_reporter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features_path = root / "features.csv"
            self._write_features(features_path)
            config_path = root / "shared.yaml"
            config_path.write_text(f"features:\n  features_index_path: {features_path}\n")

            comparison = train_pure_geometry(
                config_path, root / "output", FakeNestedEvaluator(), FakeGeometryReporter()
            )

            self.assertEqual(comparison[0]["mae_kg"], "1")

    def test_main_prints_comparison_and_returns_success(self) -> None:
        comparison = [{"model": "ridge", "mae_kg": "12.5", "r2": "0.75"}]
        with patch("buffalo_weight.train_pure_geometry.train_pure_geometry", return_value=comparison):
            self.assertEqual(main([]), 0)

    def test_main_returns_error_for_invalid_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text("features: []\n")

            self.assertEqual(main(["--shared-config", str(path)]), 1)

    def _write_features(self, path: Path) -> None:
        rows = []
        for index in range(1, 51):
            values = {feature: str(index) for feature in PURE_GEOMETRY_FEATURES}
            rows.append({"file_name": f"mask-{index}", "weight": str(index), **values})
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
