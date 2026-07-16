from __future__ import annotations

import argparse
import sys
from pathlib import Path

from buffalo_weight.config import load_config
from buffalo_weight.artifact_provenance import training_lock
from buffalo_weight.models import ModelConfig, parse_model_configs, validate_unique_model_configs
from buffalo_weight.train import write_model_comparison_from_outputs
from buffalo_weight.train_classical import train_classical
from buffalo_weight.train_cnn_mask import train_cnn_mask
from buffalo_weight.validation import validate_mask_files


def train_pipeline(
    shared_config_path: Path,
    classical_models_config_path: Path,
    cnn_mask_models_config_path: Path,
    device: str = "auto",
    dry_run: bool = False,
) -> None:
    shared_config = load_config(shared_config_path)
    training = shared_config.get("training")
    if not isinstance(training, dict):
        raise ValueError("shared config training section must be a map")
    classical_configs = parse_model_configs(load_config(classical_models_config_path))
    mask_configs = parse_model_configs(load_config(cnn_mask_models_config_path))
    validate_unique_model_configs([*classical_configs, *mask_configs])
    output_dir = Path(str(training["output_dir"]))
    if dry_run:
        _run_pipeline(shared_config_path, classical_models_config_path, cnn_mask_models_config_path, device, True)
        return
    with training_lock(output_dir):
        _run_pipeline(shared_config_path, classical_models_config_path, cnn_mask_models_config_path, device, False)
        write_model_comparison_from_outputs(output_dir, [*classical_configs, *cnn_configs(cnn_mask_models_config_path)])


def _run_pipeline(
    shared_config_path: Path, classical_path: Path, mask_path: Path, device: str, dry_run: bool
) -> None:
    shared_config = load_config(shared_config_path)
    validate_mask_files(shared_config)
    train_classical(shared_config_path, classical_path, dry_run)
    train_cnn_mask(shared_config_path, mask_path, device, dry_run)


def cnn_configs(path: Path) -> list[ModelConfig]:
    return parse_model_configs(load_config(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-config", required=True)
    parser.add_argument("--classical-models-config", required=True)
    parser.add_argument("--cnn-mask-models-config", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        train_pipeline(
            Path(args.shared_config),
            Path(args.classical_models_config),
            Path(args.cnn_mask_models_config),
            args.device,
            args.dry_run,
        )
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
