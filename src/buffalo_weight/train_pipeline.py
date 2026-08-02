from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from buffalo_weight.config import load_config
from buffalo_weight.artifact_provenance import training_lock
from buffalo_weight.models import ModelConfig, parse_model_configs, validate_unique_model_configs
from buffalo_weight.neural_cli import add_neural_execution_arguments
from buffalo_weight.neural_environment import OFFICIAL_NEURAL_DEVICE
from buffalo_weight.environment_contract import RuntimeProbe
from buffalo_weight.train import write_model_comparison_from_outputs
from buffalo_weight.train_classical import train_classical
from buffalo_weight.train_cnn_mask import execute_cuda_validated_cnn_mask_training
from buffalo_weight.system_setup import require_official_neural_runtime
from buffalo_weight.validation import validate_mask_files


@dataclass(frozen=True)
class _TrainingPipelineRequest:
    shared_config_path: Path
    classical_models_config_path: Path
    cnn_mask_models_config_path: Path
    device: str
    dry_run: bool


def train_pipeline(
    shared_config_path: Path,
    classical_models_config_path: Path,
    cnn_mask_models_config_path: Path,
    device: str = OFFICIAL_NEURAL_DEVICE,
    dry_run: bool = False, runtime_probe: RuntimeProbe | None = None,
) -> None:
    """Run the full training funnel; for example, ``dry_run=True`` only audits work."""
    require_official_neural_runtime(dry_run, runtime_probe)
    request = _TrainingPipelineRequest(
        shared_config_path, classical_models_config_path, cnn_mask_models_config_path, device,
        dry_run,
    )
    model_configs = _load_pipeline_configs(classical_models_config_path, cnn_mask_models_config_path)
    if dry_run:
        _run_pipeline(request)
        return
    _run_locked_pipeline(request, model_configs)


def _run_locked_pipeline(
    request: _TrainingPipelineRequest, model_configs: list[ModelConfig]
) -> None:
    output_dir = _training_output_dir(request.shared_config_path)
    with training_lock(output_dir):
        _run_pipeline(request)
        write_model_comparison_from_outputs(output_dir, model_configs)


def _load_pipeline_configs(classical_path: Path, mask_path: Path) -> list[ModelConfig]:
    classical_configs = parse_model_configs(load_config(classical_path))
    mask_configs = cnn_configs(mask_path)
    model_configs = [*classical_configs, *mask_configs]
    validate_unique_model_configs(model_configs)
    return model_configs


def _training_output_dir(shared_config_path: Path) -> Path:
    training = load_config(shared_config_path).get("training")
    if isinstance(training, dict):
        return Path(str(training["output_dir"]))
    raise ValueError(
        f"shared config training section was {training!r}; expected a map"
    )


def _run_pipeline(request: _TrainingPipelineRequest) -> None:
    shared_config = load_config(request.shared_config_path)
    validate_mask_files(shared_config)
    train_classical(
        request.shared_config_path, request.classical_models_config_path, request.dry_run
    )
    execute_cuda_validated_cnn_mask_training(
        request.shared_config_path,
        request.cnn_mask_models_config_path,
        request.device,
        request.dry_run,
    )


def cnn_configs(path: Path) -> list[ModelConfig]:
    """Load comparison configurations.

    Example: pass the official mask-model YAML path.
    """
    return parse_model_configs(load_config(path))


def main(
    argv: list[str] | None = None,
    runtime_probe: RuntimeProbe | None = None,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run the training CLI; for example, ``main([..., "--dry-run"])`` skips CUDA."""
    args = _build_parser().parse_args(argv)
    try:
        return _run_training_command(args, runtime_probe)
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(error, file=stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-config", required=True)
    parser.add_argument("--classical-models-config", required=True)
    parser.add_argument("--cnn-mask-models-config", required=True)
    add_neural_execution_arguments(parser)
    return parser


def _run_training_command(
    args: argparse.Namespace, runtime_probe: RuntimeProbe | None
) -> int:
    train_pipeline(
        Path(args.shared_config),
        Path(args.classical_models_config),
        Path(args.cnn_mask_models_config),
        args.device,
        args.dry_run,
        runtime_probe,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
