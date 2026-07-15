from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from buffalo_weight.config import load_config


COMPARISON_FILES = ("model_comparison.csv", "model_comparison.png")


def parse_model_names(values: list[str]) -> list[str]:
    names = []
    for value in values:
        for name in value.split(","):
            stripped = name.strip()
            if stripped and stripped not in names:
                names.append(stripped)
    return names


def _validate_cleanup_directory(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    forbidden = {Path("/").resolve(), Path.cwd().resolve(), Path.home().resolve()}
    if resolved in forbidden:
        raise ValueError(f"training output directory was {resolved}; expected a dedicated subdirectory")


def clean_training_outputs(output_dir: Path, model_names: list[str]) -> None:
    _validate_cleanup_directory(output_dir)
    if not model_names:
        shutil.rmtree(output_dir, ignore_errors=True)
        return
    for model_name in model_names:
        if Path(model_name).name != model_name or model_name in {".", ".."}:
            raise ValueError(f"model config name was {model_name!r}; expected a single directory name")
        shutil.rmtree(output_dir / model_name, ignore_errors=True)
    for file_name in COMPARISON_FILES:
        (output_dir / file_name).unlink(missing_ok=True)


def clean_from_shared_config(shared_config_path: Path, model_names: list[str]) -> None:
    shared_config = load_config(shared_config_path)
    training = shared_config["training"]
    if not isinstance(training, dict):
        raise ValueError(f"shared config training section was {training!r}; expected a map")
    clean_training_outputs(Path(str(training["output_dir"])), model_names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-config", required=True)
    parser.add_argument("--models", nargs="*", default=[])
    args = parser.parse_args(argv)
    model_names = parse_model_names(args.models)
    try:
        clean_from_shared_config(Path(args.shared_config), model_names)
    except (KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    cleaned = ", ".join(model_names) if model_names else "all training outputs"
    print(f"Cleaned {cleaned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
