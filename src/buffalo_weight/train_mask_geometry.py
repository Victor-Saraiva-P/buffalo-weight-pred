from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from buffalo_weight.artifact_provenance import (
    TrainingEvidence,
    prepare_artifacts,
    print_artifact_plan,
    training_lock,
)
from buffalo_weight.config import load_config
from buffalo_weight.mask_geometry_reports import write_mask_geometry_reports
from buffalo_weight.models import CNN_MASK_GEOMETRY_MODEL, CNN_MASK_MODEL, ModelConfig, parse_model_configs
from buffalo_weight.pure_geometry_evaluation import (
    PURE_GEOMETRY_FEATURES,
    load_pure_geometry_rows,
    stratified_geometry_rows,
)
from buffalo_weight.train import evaluate_models


EvaluationFunction = Callable[
    [list[dict[str, str]], list[str], list[ModelConfig], Path, str],
    tuple[list[dict[str, str]], list[dict[str, str]]],
]
ReportFunction = Callable[
    [
        list[dict[str, str]],
        list[dict[str, str]],
        list[ModelConfig],
        TrainingEvidence,
        Path,
    ],
    list[dict[str, str]],
]


def train_mask_geometry_comparison(
    shared_config_path: Path, models_config_path: Path, output_dir: Path,
    evaluator: EvaluationFunction = evaluate_models,
    reporter: ReportFunction = write_mask_geometry_reports,
    device: str = "auto",
) -> list[dict[str, str]]:
    """Compare mask-only and pure-geometry fusion; for example, ``train_mask_geometry_comparison(...)``."""
    shared_config = load_config(shared_config_path)
    model_configs = _comparison_configs(models_config_path)
    rows, masks_dir = _comparison_inputs(shared_config)
    features = list(PURE_GEOMETRY_FEATURES)
    evidence = TrainingEvidence(rows, rows, features, masks_dir, device)
    plans, pending_configs = prepare_artifacts(output_dir, model_configs, evidence)
    print_artifact_plan(plans)
    if pending_configs:
        (output_dir / "report.md").unlink(missing_ok=True)
        metrics, predictions = evaluator(rows, features, pending_configs, masks_dir, device)
    else:
        metrics, predictions = [], []
    return reporter(metrics, predictions, model_configs, evidence, output_dir)


def _comparison_configs(path: Path) -> list[ModelConfig]:
    configs = parse_model_configs(load_config(path))
    _validate_comparison_models(configs)
    return configs


def _comparison_inputs(config: dict[object, object]) -> tuple[list[dict[str, str]], Path]:
    features = config.get("features")
    data = config.get("data")
    if not isinstance(features, dict):
        raise ValueError(f"config features was {features!r}; expected a map")
    if not isinstance(data, dict):
        raise ValueError(f"config data was {data!r}; expected a map")
    rows = load_pure_geometry_rows(Path(str(features.get("features_index_path"))))
    return stratified_geometry_rows(rows, 5, 10, 42), Path(str(data.get("masks_dir")))


def _validate_comparison_models(configs: list[ModelConfig]) -> None:
    models = [config.model for config in configs]
    expected = [CNN_MASK_MODEL, CNN_MASK_GEOMETRY_MODEL]
    if sorted(models) == sorted(expected):
        return
    raise ValueError(f"comparison models were {models!r}; expected exactly {expected!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-config", default="configs/shared.yaml")
    parser.add_argument("--models-config", default="configs/cnn_mask_geometry_experiment.yaml")
    parser.add_argument("--output-dir", default="generated/mask_geometry")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    try:
        output_dir = Path(args.output_dir)
        with training_lock(output_dir):
            comparison = train_mask_geometry_comparison(
                Path(args.shared_config), Path(args.models_config), output_dir, device=args.device
            )
    except (KeyError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    for row in comparison:
        print(f"{row['model']}: MAE={float(row['mae_kg']):.2f} kg, R2={float(row['r2']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
