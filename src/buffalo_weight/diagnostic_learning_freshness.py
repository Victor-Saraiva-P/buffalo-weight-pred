"""Freshness and reusability checks for 100% baseline point artifacts.

Reference: GitHub Issue #25.
"""

from __future__ import annotations

import csv
import json
import numpy as np

from buffalo_weight.reproduction_config import ReportContract


def check_baseline_100_reusability(
    contract: ReportContract,
    configuration: str,
) -> bool:
    """Check if 100% baseline evaluation predictions are current and reusable.

    Example: ``check_baseline_100_reusability(contract, "random_forest_baseline")`` returns True or False.
    """
    output_dir = contract.artifacts_root / "baselines" / configuration
    manifest_path = output_dir / "manifest.json"
    predictions_path = output_dir / "predictions.csv"

    if not manifest_path.is_file() or not predictions_path.is_file():
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("status") != "complete":
            return False
    except (OSError, ValueError, json.JSONDecodeError):
        return False

    return True


def load_reused_fold_metrics(
    contract: ReportContract,
    configuration: str,
    fold: int,
) -> tuple[float, float, int]:
    """Load MAE, bias, and n_eval from existing 100% baseline predictions CSV for a fold.

    Example: ``load_reused_fold_metrics(contract, "random_forest_baseline", fold=1)`` returns (mae, bias, n_eval).
    """
    predictions_path = contract.artifacts_root / "baselines" / configuration / "predictions.csv"
    if not predictions_path.is_file():
        raise ValueError(f"predictions.csv was missing at {predictions_path} for configuration {configuration}")

    observed: list[float] = []
    predicted: list[float] = []

    with predictions_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["fold"]) == fold:
                observed.append(float(row["observed_weight_kg"]))
                predicted.append(float(row["predicted_weight_kg"]))

    if not observed:
        raise ValueError(f"predictions.csv at {predictions_path} contained no rows for fold {fold}")

    obs_arr = np.asarray(observed, dtype=np.float64)
    pred_arr = np.asarray(predicted, dtype=np.float64)
    diff = pred_arr - obs_arr

    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))

    return mae, bias, len(observed)
