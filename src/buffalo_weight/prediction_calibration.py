from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from buffalo_weight.csv_io import write_csv_rows
from buffalo_weight.diagnostic_metrics import metric_summary
from buffalo_weight.split import read_rows
from buffalo_weight.train import format_metric


CALIBRATION_FIELDS = ["method", "n", "mae", "median_ae", "rmse", "bias", "p90_ae", "r2", "b10_mae", "b10_bias"]


def cross_fitted_calibration(rows: list[dict[str, str]], method: str) -> np.ndarray:
    """Calibrate each fold without its labels; for example, ``cross_fitted_calibration(rows, "linear")``."""
    actual = np.asarray([float(row["weight"]) for row in rows])
    predicted = np.asarray([float(row["y_pred"]) for row in rows])
    folds = np.asarray([int(row["fold"]) for row in rows])
    calibrated = np.zeros(len(rows))
    for fold in sorted(set(folds)):
        train, validation = folds != fold, folds == fold
        calibrated[validation] = _fit_calibrator(predicted[train], actual[train], method, predicted[validation])
    return calibrated


def _fit_calibrator(predicted: np.ndarray, actual: np.ndarray, method: str, validation: np.ndarray) -> np.ndarray:
    if method == "linear":
        return LinearRegression().fit(predicted[:, None], actual).predict(validation[:, None])
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(predicted, actual).predict(validation)
    raise ValueError(f"calibration method was {method!r}; expected 'linear' or 'isotonic'")


def calibration_summary(rows: list[dict[str, str]], method: str, predicted: np.ndarray) -> dict[str, str]:
    """Summarize calibrated OOF predictions; for example, ``calibration_summary(rows, "linear", p)``."""
    actual = np.asarray([float(row["weight"]) for row in rows])
    categories = np.asarray([row["weight_category"] for row in rows])
    heavy = categories == "B10"
    heavy_errors = predicted[heavy] - actual[heavy]
    return {
        "method": method,
        **metric_summary(actual, predicted),
        "b10_mae": format_metric(float(np.abs(heavy_errors).mean())),
        "b10_bias": format_metric(float(heavy_errors.mean())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="generated/calibration/model_comparison.csv")
    args = parser.parse_args(argv)
    rows = read_rows(Path(args.predictions))
    actual_prediction = np.asarray([float(row["y_pred"]) for row in rows])
    summaries = [calibration_summary(rows, "uncalibrated", actual_prediction)]
    for method in ("linear", "isotonic"):
        summaries.append(calibration_summary(rows, method, cross_fitted_calibration(rows, method)))
    write_csv_rows(summaries, Path(args.output), CALIBRATION_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
