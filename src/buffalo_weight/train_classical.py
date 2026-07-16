from __future__ import annotations

import argparse
import sys
from pathlib import Path

from buffalo_weight.config import load_config
from buffalo_weight.artifact_provenance import TrainingEvidence, prepare_artifacts
from buffalo_weight.artifact_provenance import print_artifact_plan, training_lock
from buffalo_weight.models import MASK_PREDICTION_MODELS, ModelConfig, parse_model_configs
from buffalo_weight.split import read_rows
from buffalo_weight.train import evaluate_models, join_rows, write_training_outputs
from buffalo_weight.validation import validate_feature_index, validate_split


def train_classical(
    shared_config_path: Path, models_config_path: Path, dry_run: bool = False
) -> list[ModelConfig]:
    shared_config = load_config(shared_config_path)
    models_config = load_config(models_config_path)
    model_configs = parse_model_configs(models_config)
    unsupported = [config.name for config in model_configs if config.model in MASK_PREDICTION_MODELS]
    if unsupported:
        raise ValueError("train_classical only supports classical prediction models")
    feature_columns = models_config["feature_columns"]
    if not isinstance(feature_columns, list):
        raise ValueError("classical models config feature_columns must be a list")

    validate_split(shared_config)
    validate_feature_index(shared_config, [str(column) for column in feature_columns])
    features = shared_config["features"]
    split = shared_config["split"]
    training = shared_config["training"]
    data = shared_config["data"]
    if not isinstance(features, dict):
        raise ValueError("shared config features section must be a map")
    if not isinstance(split, dict):
        raise ValueError("shared config split section must be a map")
    if not isinstance(training, dict):
        raise ValueError("shared config training section must be a map")
    if not isinstance(data, dict):
        raise ValueError("shared config data section must be a map")

    feature_rows = read_rows(Path(str(features["features_index_path"])))
    split_rows = read_rows(Path(str(split["split_path"])))
    rows = join_rows(feature_rows, split_rows)
    output_dir = Path(str(training["output_dir"]))
    evidence = TrainingEvidence(
        split_rows,
        feature_rows,
        [str(column) for column in feature_columns],
        Path(str(data["masks_dir"])),
        "auto",
    )
    plans, pending_configs = prepare_artifacts(output_dir, model_configs, evidence, dry_run)
    print_artifact_plan(plans)
    if dry_run:
        return model_configs
    if pending_configs:
        metrics, predictions = evaluate_models(
            rows,
            [str(column) for column in feature_columns],
            pending_configs,
            Path(str(data["masks_dir"])),
        )
        write_training_outputs(output_dir, pending_configs, metrics, predictions, evidence)
    skipped = [config.name for config in model_configs if config not in pending_configs]
    if skipped:
        print(f"Skipping completed model configs: {', '.join(skipped)}")
    return model_configs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-config", required=True)
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        shared_config = load_config(Path(args.shared_config))
        training = shared_config["training"]
        if not isinstance(training, dict):
            raise ValueError("shared config training section must be a map")
        output_dir = Path(str(training["output_dir"]))
        if args.dry_run:
            train_classical(Path(args.shared_config), Path(args.models_config), True)
        else:
            with training_lock(output_dir):
                train_classical(Path(args.shared_config), Path(args.models_config))
    except (KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
