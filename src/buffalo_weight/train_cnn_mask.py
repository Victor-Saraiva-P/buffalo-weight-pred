from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from buffalo_weight.config import load_config
from buffalo_weight.artifact_provenance import TrainingEvidence, prepare_artifacts
from buffalo_weight.artifact_provenance import print_artifact_plan, training_lock
from buffalo_weight.models import MASK_PREDICTION_MODELS, ModelConfig, parse_model_configs
from buffalo_weight.neural_environment import (
    OFFICIAL_NEURAL_DEVICE,
    OFFICIAL_NEURAL_DEVICE_CHOICES,
    require_neural_cuda,
)
from buffalo_weight.report_environment import RuntimeProbe
from buffalo_weight.split import read_rows
from buffalo_weight.system_setup import SystemRuntimeProbe
from buffalo_weight.train import evaluate_models, write_training_outputs
from buffalo_weight.validation import validate_mask_files, validate_split


@dataclass(frozen=True)
class _MaskTrainingInputs:
    rows: list[dict[str, str]]
    masks_dir: Path
    output_dir: Path


def train_cnn_mask(
    shared_config_path: Path,
    models_config_path: Path,
    device: str = OFFICIAL_NEURAL_DEVICE,
    dry_run: bool = False,
) -> list[ModelConfig]:
    """Train mask predictors; for example, pass ``dry_run=True`` to audit artifacts."""
    shared_config = load_config(shared_config_path)
    model_configs = _load_mask_model_configs(models_config_path)
    inputs = _load_mask_training_inputs(shared_config)
    evidence = TrainingEvidence(inputs.rows, [], [], inputs.masks_dir, device)
    plans, pending_configs = prepare_artifacts(
        inputs.output_dir, model_configs, evidence, dry_run
    )
    print_artifact_plan(plans)
    if dry_run:
        return model_configs
    _evaluate_pending_models(inputs, pending_configs, evidence, device)
    _print_skipped_models(model_configs, pending_configs)
    return model_configs


def _load_mask_model_configs(path: Path) -> list[ModelConfig]:
    model_configs = parse_model_configs(load_config(path))
    unsupported = [
        config.name for config in model_configs if config.model not in MASK_PREDICTION_MODELS
    ]
    if not unsupported:
        return model_configs
    raise ValueError(
        f"model configs were {unsupported!r}; expected train_cnn_mask only supports "
        f"mask prediction models {sorted(MASK_PREDICTION_MODELS)!r}"
    )


def _load_mask_training_inputs(config: dict[str, object]) -> _MaskTrainingInputs:
    validate_mask_files(config)
    validate_split(config)
    data_section = _require_mapping_section(config, "data")
    split_section = _require_mapping_section(config, "split")
    training_section = _require_mapping_section(config, "training")
    rows = read_rows(Path(str(split_section["split_path"])))
    return _MaskTrainingInputs(
        rows,
        Path(str(data_section["masks_dir"])),
        Path(str(training_section["output_dir"])),
    )


def _require_mapping_section(
    config: dict[str, object], section_name: str
) -> dict[object, object]:
    section = config.get(section_name)
    if isinstance(section, dict):
        return section
    raise ValueError(
        f"shared config {section_name} section was {section!r}; expected a map"
    )


def _evaluate_pending_models(
    inputs: _MaskTrainingInputs,
    pending_configs: list[ModelConfig],
    evidence: TrainingEvidence,
    device: str,
) -> None:
    if not pending_configs:
        return
    metrics, predictions = evaluate_models(
        inputs.rows, [], pending_configs, inputs.masks_dir, device
    )
    write_training_outputs(
        inputs.output_dir, pending_configs, metrics, predictions, evidence
    )


def _print_skipped_models(
    model_configs: list[ModelConfig], pending_configs: list[ModelConfig]
) -> None:
    skipped = [config.name for config in model_configs if config not in pending_configs]
    if not skipped:
        return
    print(f"Skipping completed model configs: {', '.join(skipped)}")


def main(
    argv: list[str] | None = None,
    runtime_probe: RuntimeProbe | None = None,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run mask training; for example, ``main([..., "--dry-run"])`` avoids CUDA work."""
    args = _build_parser().parse_args(argv)
    try:
        return _run_training_command(args, runtime_probe)
    except (KeyError, ValueError) as error:
        print(error, file=stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-config", required=True)
    parser.add_argument("--models-config", required=True)
    parser.add_argument(
        "--device", choices=OFFICIAL_NEURAL_DEVICE_CHOICES, default=OFFICIAL_NEURAL_DEVICE
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run_training_command(
    args: argparse.Namespace, runtime_probe: RuntimeProbe | None
) -> int:
    if args.dry_run:
        train_cnn_mask(
            Path(args.shared_config), Path(args.models_config), args.device, True
        )
        return 0
    require_neural_cuda((runtime_probe or SystemRuntimeProbe()).compute_environment())
    shared_config = load_config(Path(args.shared_config))
    training = _require_mapping_section(shared_config, "training")
    with training_lock(Path(str(training["output_dir"]))):
        train_cnn_mask(Path(args.shared_config), Path(args.models_config), args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
