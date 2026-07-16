from __future__ import annotations

import argparse
import sys
from pathlib import Path

from buffalo_weight.config import load_config
from buffalo_weight.artifact_provenance import TrainingEvidence, prepare_artifacts
from buffalo_weight.artifact_provenance import print_artifact_plan, training_lock
from buffalo_weight.models import MASK_PREDICTION_MODELS, ModelConfig, parse_model_configs
from buffalo_weight.split import read_rows
from buffalo_weight.train import evaluate_models, write_training_outputs
from buffalo_weight.validation import validate_mask_files, validate_split


def train_cnn_mask(
    shared_config_path: Path,
    models_config_path: Path,
    device: str = "auto",
    dry_run: bool = False,
) -> list[ModelConfig]:
    shared_config = load_config(shared_config_path)
    models_config = load_config(models_config_path)
    model_configs = parse_model_configs(models_config)
    unsupported = [config.name for config in model_configs if config.model not in MASK_PREDICTION_MODELS]
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
    output_dir = Path(str(training["output_dir"]))
    evidence = TrainingEvidence(rows, [], [], Path(str(data["masks_dir"])), device)
    plans, pending_configs = prepare_artifacts(output_dir, model_configs, evidence, dry_run)
    print_artifact_plan(plans)
    if dry_run:
        return model_configs
    if pending_configs:
        metrics, predictions = evaluate_models(
            rows,
            feature_columns=[],
            model_configs=pending_configs,
            masks_dir=Path(str(data["masks_dir"])),
            device=device,
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        shared_config = load_config(Path(args.shared_config))
        training = shared_config["training"]
        if not isinstance(training, dict):
            raise ValueError("shared config training section must be a map")
        output_dir = Path(str(training["output_dir"]))
        if args.dry_run:
            train_cnn_mask(Path(args.shared_config), Path(args.models_config), args.device, True)
        else:
            with training_lock(output_dir):
                train_cnn_mask(Path(args.shared_config), Path(args.models_config), args.device)
    except (KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
