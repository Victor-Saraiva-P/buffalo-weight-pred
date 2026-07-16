from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from buffalo_weight.csv_io import write_csv_rows
from buffalo_weight.split import read_rows
from buffalo_weight.train import format_metric


COMPARISON_FIELDS = [
    "ensemble",
    "model_count",
    "mae",
    "rmse",
    "r2",
    "fold_mae_mean",
    "fold_mae_std",
]


def _index_rows(rows: list[dict[str, str]], model_name: str) -> dict[str, dict[str, str]]:
    indexed = {row["file_name"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"predictions for {model_name} contain duplicate file_name values; expected unique rows")
    return indexed


def _aligned_arrays(
    predictions_by_model: dict[str, list[dict[str, str]]], names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indexed = {name: _index_rows(predictions_by_model[name], name) for name in names}
    file_names = sorted(indexed[names[0]])
    expected = set(file_names)
    for name in names[1:]:
        if set(indexed[name]) != expected:
            raise ValueError(f"predictions for {name} do not match {names[0]}; expected identical file_name values")
    reference = indexed[names[0]]
    for name in names[1:]:
        for file_name in file_names:
            reference_pair = (reference[file_name]["weight"], reference[file_name]["fold"])
            candidate_pair = (indexed[name][file_name]["weight"], indexed[name][file_name]["fold"])
            if candidate_pair != reference_pair:
                raise ValueError(
                    f"prediction for {file_name} in {name} was {candidate_pair}; expected {reference_pair}"
                )
    actual = np.asarray([float(reference[file_name]["weight"]) for file_name in file_names])
    folds = np.asarray([int(reference[file_name]["fold"]) for file_name in file_names])
    matrix = np.asarray([[float(indexed[name][file_name]["y_pred"]) for name in names] for file_name in file_names])
    return actual, folds, matrix


def _summary(
    names: tuple[str, ...], actual: np.ndarray, folds: np.ndarray, predicted: np.ndarray
) -> dict[str, str]:
    fold_maes = [mean_absolute_error(actual[folds == fold], predicted[folds == fold]) for fold in sorted(set(folds))]
    return {
        "ensemble": "+".join(names),
        "model_count": str(len(names)),
        "mae": format_metric(mean_absolute_error(actual, predicted)),
        "rmse": format_metric(mean_squared_error(actual, predicted) ** 0.5),
        "r2": format_metric(r2_score(actual, predicted)),
        "fold_mae_mean": format_metric(float(np.mean(fold_maes))),
        "fold_mae_std": format_metric(float(np.std(fold_maes))),
    }


def evaluate_equal_weight_ensembles(
    predictions_by_model: dict[str, list[dict[str, str]]], max_model_count: int = 3
) -> list[dict[str, str]]:
    """Compare equal-weight OOF ensembles; for example, ``evaluate_equal_weight_ensembles(predictions)``."""
    if not predictions_by_model or max_model_count < 1:
        raise ValueError(
            f"ensemble inputs were models={len(predictions_by_model)}, max_model_count={max_model_count}; "
            "expected at least one model and max_model_count >= 1"
        )
    names = sorted(predictions_by_model)
    summaries = []
    for count in range(1, min(max_model_count, len(names)) + 1):
        for selected in combinations(names, count):
            actual, folds, matrix = _aligned_arrays(predictions_by_model, selected)
            summaries.append(_summary(selected, actual, folds, matrix.mean(axis=1)))
    return sorted(summaries, key=lambda row: float(row["mae"]))


def load_model_predictions(output_dir: Path, model_names: list[str]) -> dict[str, list[dict[str, str]]]:
    """Load model OOF CSV files; for example, ``load_model_predictions(Path("train"), ["rf"])``."""
    return {name: read_rows(output_dir / name / "predictions.csv") for name in model_names}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output-dir", default="generated/train")
    parser.add_argument("--models", required=True)
    parser.add_argument("--max-model-count", type=int, default=3)
    parser.add_argument("--output", default="generated/ensemble/model_comparison.csv")
    args = parser.parse_args(argv)
    try:
        names = [name.strip() for name in args.models.split(",") if name.strip()]
        predictions = load_model_predictions(Path(args.train_output_dir), names)
        rows = evaluate_equal_weight_ensembles(predictions, args.max_model_count)
        write_csv_rows(rows, Path(args.output), COMPARISON_FIELDS)
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
