from __future__ import annotations

import argparse
import sys
from pathlib import Path

from buffalo_weight.config import load_config
from buffalo_weight.models import CNN_MASK_MODEL, ModelConfig, parse_model_configs
from buffalo_weight.split import read_rows
from buffalo_weight.train import evaluate_models, write_training_outputs
from buffalo_weight.validation import validate_mask_files, validate_split


def train_cnn_mask(shared_config_path: Path, models_config_path: Path) -> list[ModelConfig]:
    shared_config = load_config(shared_config_path)
    models_config = load_config(models_config_path)
    model_configs = parse_model_configs(models_config)
    unsupported = [config.name for config in model_configs if config.model != CNN_MASK_MODEL]
    if unsupported:
        raise ValueError("train_cnn_mask only supports mask prediction models")

    validate_mask_files(shared_config)
    validate_split(shared_config)
    data = shared_config["data"]
    split = shared_config["split"]
    training = shared_config["training"]
    if not isinstance(data, dict):
        raise ValueError("shared config data section must be a map")
    if not isinstance(split, dict):
        raise ValueError("shared config split section must be a map")
    if not isinstance(training, dict):
        raise ValueError("shared config training section must be a map")

    rows = read_rows(Path(str(split["split_path"])))
    metrics, predictions = evaluate_models(
        rows,
        feature_columns=[],
        model_configs=model_configs,
        masks_dir=Path(str(data["masks_dir"])),
    )
    write_training_outputs(Path(str(training["output_dir"])), model_configs, metrics, predictions)
    return model_configs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-config", required=True)
    parser.add_argument("--models-config", required=True)
    args = parser.parse_args(argv)

    try:
        train_cnn_mask(Path(args.shared_config), Path(args.models_config))
    except (KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
