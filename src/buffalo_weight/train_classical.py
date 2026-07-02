from __future__ import annotations

import argparse
import sys
from pathlib import Path

from buffalo_weight.config import load_config
from buffalo_weight.models import CNN_MASK_MODEL, ModelConfig, parse_model_configs
from buffalo_weight.split import read_rows
from buffalo_weight.train import evaluate_models, join_rows, write_training_outputs
from buffalo_weight.validation import validate_feature_index, validate_split


def train_classical(shared_config_path: Path, models_config_path: Path) -> list[ModelConfig]:
    shared_config = load_config(shared_config_path)
    models_config = load_config(models_config_path)
    model_configs = parse_model_configs(models_config)
    unsupported = [config.name for config in model_configs if config.model == CNN_MASK_MODEL]
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
    if not isinstance(features, dict):
        raise ValueError("shared config features section must be a map")
    if not isinstance(split, dict):
        raise ValueError("shared config split section must be a map")
    if not isinstance(training, dict):
        raise ValueError("shared config training section must be a map")

    feature_rows = read_rows(Path(str(features["features_index_path"])))
    split_rows = read_rows(Path(str(split["split_path"])))
    rows = join_rows(feature_rows, split_rows)
    metrics, predictions = evaluate_models(rows, [str(column) for column in feature_columns], model_configs)
    write_training_outputs(Path(str(training["output_dir"])), model_configs, metrics, predictions)
    return model_configs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-config", required=True)
    parser.add_argument("--models-config", required=True)
    args = parser.parse_args(argv)

    try:
        train_classical(Path(args.shared_config), Path(args.models_config))
    except (KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
