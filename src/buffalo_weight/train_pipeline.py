from __future__ import annotations

import argparse
import sys
from pathlib import Path

from buffalo_weight.config import load_config
from buffalo_weight.train import write_model_comparison_from_outputs
from buffalo_weight.train_classical import train_classical
from buffalo_weight.train_cnn_mask import train_cnn_mask
from buffalo_weight.validation import validate_mask_files


def train_pipeline(
    shared_config_path: Path,
    classical_models_config_path: Path,
    cnn_mask_models_config_path: Path,
    device: str = "auto",
) -> None:
    shared_config = load_config(shared_config_path)
    validate_mask_files(shared_config)
    classical_configs = train_classical(shared_config_path, classical_models_config_path)
    cnn_mask_configs = train_cnn_mask(shared_config_path, cnn_mask_models_config_path, device)
    training = shared_config["training"]
    if not isinstance(training, dict):
        raise ValueError("shared config training section must be a map")
    write_model_comparison_from_outputs(
        Path(str(training["output_dir"])),
        [*classical_configs, *cnn_mask_configs],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-config", required=True)
    parser.add_argument("--classical-models-config", required=True)
    parser.add_argument("--cnn-mask-models-config", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)

    try:
        train_pipeline(
            Path(args.shared_config),
            Path(args.classical_models_config),
            Path(args.cnn_mask_models_config),
            args.device,
        )
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
