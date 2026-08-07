"""Freshness and reusability checks for 100% baseline point artifacts.

Reference: GitHub Issue #25.
"""

from __future__ import annotations

import csv
import json
import numpy as np

from buffalo_weight.baseline_manifest import baseline_configuration_status
from buffalo_weight.baseline_provenance import BaselineProvenance, SystemBaselineProvenance
from buffalo_weight.compact_cnn_manifest import compact_cnn_status
from buffalo_weight.compact_cnn_provenance import CompactCnnProvenance, SystemCompactCnnProvenance
from buffalo_weight.compact_cnn_types import COMPACT_CNN_RECIPE
from buffalo_weight.dense_baseline_manifest import dense_baseline_status
from buffalo_weight.feature_confirmation_manifest import validate_frozen_feature_contract
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.resnet_baseline_provenance import ResNetBaselineProvenance, SystemResNetBaselineProvenance
from buffalo_weight.resnet_baseline_stage import resnet_baseline_status


def check_baseline_100_reusability(
    contract: ReportContract,
    configuration: str,
    baseline_provenance: BaselineProvenance | None = None,
    compact_provenance: CompactCnnProvenance | None = None,
    resnet_provenance: ResNetBaselineProvenance | None = None,
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
        status = _determine_configuration_status(
            contract, configuration, baseline_provenance, compact_provenance, resnet_provenance
        )
        return status == "reusable"
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _determine_configuration_status(
    contract: ReportContract,
    configuration: str,
    baseline_provenance: BaselineProvenance | None,
    compact_provenance: CompactCnnProvenance | None,
    resnet_provenance: ResNetBaselineProvenance | None,
) -> str:
    if configuration == "random_forest_baseline":
        features = validate_frozen_feature_contract(contract)
        prov = baseline_provenance or SystemBaselineProvenance()
        return baseline_configuration_status(contract, "random_forest_baseline", "candidate", features, prov)
    if configuration == "dense_baseline":
        return dense_baseline_status(contract).rsplit(": ", maxsplit=1)[-1]
    if configuration == "compact_cnn_baseline":
        compact_prov = compact_provenance or SystemCompactCnnProvenance()
        return compact_cnn_status(contract, COMPACT_CNN_RECIPE, compact_prov)
    if configuration == "resnet18_pretrained_partial":
        resnet_prov = resnet_provenance or SystemResNetBaselineProvenance()
        return resnet_baseline_status(contract, resnet_prov)
    raise ValueError(f"unknown baseline configuration {configuration!r}; expected one of four baselines")


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
        raise ValueError(f"predictions.csv missing at {predictions_path} for configuration {configuration!r}")

    observed, predicted = _read_fold_predictions(predictions_path, fold)
    if not observed:
        raise ValueError(f"predictions.csv at {predictions_path} contained 0 rows for fold {fold}")

    obs_arr = np.asarray(observed, dtype=np.float64)
    pred_arr = np.asarray(predicted, dtype=np.float64)
    diff = pred_arr - obs_arr

    return float(np.mean(np.abs(diff))), float(np.mean(diff)), len(observed)


def _read_fold_predictions(path: Path, fold: int) -> tuple[list[float], list[float]]:
    observed: list[float] = []
    predicted: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["fold"]) == fold:
                obs_val = row.get("observed_weight_kg") or row.get("weight_kg")
                pred_val = row.get("predicted_weight_kg") or row.get("prediction_kg")
                if obs_val is not None and pred_val is not None:
                    observed.append(float(obs_val))
                    predicted.append(float(pred_val))
    return observed, predicted
