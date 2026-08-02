"""Shared command-line contract for official neural execution."""

import argparse

from buffalo_weight.neural_environment import (
    OFFICIAL_NEURAL_DEVICE,
    OFFICIAL_NEURAL_DEVICE_CHOICES,
)


def add_neural_execution_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the official device contract; for example, training parsers allow only CUDA."""
    parser.add_argument(
        "--device", choices=OFFICIAL_NEURAL_DEVICE_CHOICES, default=OFFICIAL_NEURAL_DEVICE
    )
    parser.add_argument("--dry-run", action="store_true")
