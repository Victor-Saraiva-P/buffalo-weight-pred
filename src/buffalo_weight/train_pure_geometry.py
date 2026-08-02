from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from buffalo_weight.config import load_config
from buffalo_weight.pure_geometry_evaluation import (
    NestedEvaluation,
    load_pure_geometry_rows,
    nested_evaluate_models,
    stratified_geometry_rows,
)
from buffalo_weight.pure_geometry_reports import write_pure_geometry_reports


EvaluationFunction = Callable[[list[dict[str, str]], int, int], NestedEvaluation]
ReportFunction = Callable[[NestedEvaluation, list[dict[str, str]], Path], list[dict[str, str]]]


def train_pure_geometry(
    shared_config_path: Path, output_dir: Path, evaluator: EvaluationFunction = nested_evaluate_models,
    reporter: ReportFunction = write_pure_geometry_reports, random_state: int = 42,
) -> list[dict[str, str]]:
    """Run leakage-safe nested evaluation on pure geometry features.

    Example: ``train_pure_geometry(Path("configs/shared.yaml"), Path("generated/pure_geometry"))``.
    """
    config = load_config(shared_config_path)
    features = config.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"config features was {features!r}; expected a map")
    rows = load_pure_geometry_rows(Path(str(features.get("features_index_path"))))
    outer_rows = stratified_geometry_rows(rows, 5, 10, random_state)
    evaluation = evaluator(outer_rows, random_state, 4)
    return reporter(evaluation, rows, output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-config", default="configs/shared.yaml")
    parser.add_argument("--output-dir", default="generated/pure_geometry")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        comparison = train_pure_geometry(Path(args.shared_config), Path(args.output_dir), random_state=args.random_state)
    except (KeyError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    for row in comparison:
        print(f"{row['model']}: MAE={float(row['mae_kg']):.2f} kg, R2={float(row['r2']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
